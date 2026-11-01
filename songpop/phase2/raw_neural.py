"""Parent adapter and isolated raw-neural worker protocol.

The parent owns the returned D transform objects and constructs the identical
matrix for prediction and duplicate-input checks. Torch lives only in workers.
"""
import atexit
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np

from songpop.phase2 import neural_backend as backend
from songpop.phase2 import tabm
from songpop.phase2.raw_features import RawFeaturePipeline
from songpop.phase2.raw_recipes import validate
from songpop.phase2.neural_stopping import sha256, stopping_history

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "raw-neural-worker-v1"
SOURCES = (
    "songpop/__init__.py", "songpop/data.py", "songpop/features.py", "songpop/splits.py",
    "songpop/phase2/__init__.py", "songpop/phase2/encoders.py", "songpop/phase2/features.py",
    "songpop/phase2/tabm.py", "songpop/phase2/raw_features.py", "songpop/phase2/raw_splits.py",
    "songpop/phase2/neural_stopping.py", "songpop/phase2/raw_recipes.py", "songpop/phase2/neural_backend.py",
    "songpop/phase2/raw_neural.py", "songpop/phase2/neural_architectures.py",
    "songpop/phase2/neural_preprocessing.py", "songpop/phase2/neural_retrieval.py", "analysis/fast_features.py",
)
FORBIDDEN = {"torch", "lightgbm", "xgboost", "catboost", "songpop.artifacts", "songpop.phase2.models",
             "songpop.phase2.recipes", "songpop.phase2.engine"}


def sources():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def implementations():
    return {name: hashlib.sha256(Path(importlib.util.find_spec(name).origin).read_bytes()).hexdigest()
            for name in ("tabm", "rtdl_num_embeddings", "rtdl_revisiting_models", "torch")}


def policy():
    path = ROOT / "configs/phase2/p2.9/device_policy.json"
    declared = json.loads(path.read_text())
    if declared["policy_id"] != "linux_ieee_deterministic_v1":
        raise ValueError("unimplemented neural numerical policy")
    return declared, hashlib.sha256(path.read_bytes()).hexdigest()


def worker_import_check():
    bad = sorted((FORBIDDEN - {"torch"}) & set(sys.modules))
    if bad:
        raise RuntimeError(f"forbidden neural-worker imports: {bad}")
    observed = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py"):
            path = Path(filename).resolve()
            if ROOT in path.parents and path.is_file():
                observed.add(str(path.relative_to(ROOT)))
    if observed - set(SOURCES):
        raise RuntimeError(f"worker executes unbound project sources: {sorted(observed - set(SOURCES))}")
    return []


def serve(device, inp, out):
    torch = backend.configure(device)
    import tabm as tabm_package
    declared, policy_sha = policy()
    hello = {"protocol": PROTOCOL, "python": sys.version.split()[0], "device": device,
             "sources": sources(), "implementations": implementations(), "policy_sha256": policy_sha,
             "torch": str(torch.__version__), "tabm": str(tabm_package.__version__),
             "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
             "default_dtype": str(torch.get_default_dtype()), "interop_threads": torch.get_num_interop_threads(),
             "worker_environment": {k: os.environ.get(k) for k in declared["worker_environment"]},
             "forbidden_imports": worker_import_check()}
    if hello["worker_environment"] != declared["worker_environment"]:
        raise RuntimeError("worker did not start under the declared numerical environment")
    tabm.write_frame(out, hello)
    while True:
        try:
            request = tabm.read_frame(inp.fileno(), None)
        except EOFError:
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if request.get("protocol") != PROTOCOL or request.get("sources") != sources():
                    raise ValueError("worker request has a different protocol or source identity")
                recipe = validate(request["recipe"])
                if recipe["adapter"]["device"] != device:
                    raise ValueError("recipe device differs from worker device")
                op = request["op"]
                if op in ("fit", "trajectory"):
                    args = (torch, recipe, request["rows"], request["y"], request["groups"], request["context"],
                            request.get("diagnostics"))
                    result = backend.fit(*args) if op == "fit" else backend.trajectory(
                        *args, request["epochs"], request["seed"])
                elif op == "predict":
                    result = backend.predict(torch, recipe, request["states"], request["Z"], request["model_spec"])
                else:
                    raise ValueError(f"unknown raw-neural request {op!r}")
                worker_import_check()
                reply = {"ok": result}
            except Exception as error:
                reply = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        reply.update(protocol=PROTOCOL, id=request.get("id"), sources=sources(),
                     warnings=[f"{type(w.message).__name__}: {w.message}" for w in caught])
        tabm.write_frame(out, reply)


class Worker:
    workers = {}

    def __init__(self, device):
        self.device, self.proc, self.counter = device, None, 0
        self.expected_sources = sources()
        declared, policy_sha = policy()
        self.proc = subprocess.Popen([sys.executable, "-m", "songpop.phase2.raw_neural", "--worker", device],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=dict(os.environ, **declared["worker_environment"], PYTHONDONTWRITEBYTECODE="1"))
        try:
            self.hello = tabm.read_frame(self.proc.stdout.fileno(), time.monotonic() + 120)
            expected = {"protocol": PROTOCOL, "python": sys.version.split()[0], "device": device,
                        "sources": self.expected_sources, "implementations": implementations(),
                        "policy_sha256": policy_sha, "worker_environment": declared["worker_environment"],
                        "default_dtype": "torch.float32", "interop_threads": 1, "forbidden_imports": []}
            if any(self.hello.get(k) != value for k, value in expected.items()):
                raise RuntimeError("raw neural worker hello differs from the frozen source/runtime policy")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    @classmethod
    def for_device(cls, device):
        if "torch" in sys.modules:
            raise RuntimeError("Torch must never be imported into the fitting parent")
        instance = cls.workers.get(device)
        if instance is None or instance.proc is None or instance.proc.poll() is not None:
            if instance is not None:
                instance.close()
            instance = cls.workers[device] = cls(device)
        return instance

    @classmethod
    def close_all(cls):
        for worker in cls.workers.values():
            worker.close()
        cls.workers.clear()

    def call(self, request):
        if sources() != self.expected_sources or implementations() != self.hello["implementations"]:
            self.close()
            raise RuntimeError("raw-neural sources changed after worker startup")
        self.counter += 1
        rid = f"{os.getpid()}-{self.counter}"
        try:
            tabm.write_frame(self.proc.stdin, dict(request, protocol=PROTOCOL, id=rid, sources=self.expected_sources))
            reply = tabm.read_frame(self.proc.stdout.fileno(), time.monotonic() + tabm.EXCHANGE_SECONDS)
            if (reply.get("protocol") != PROTOCOL or reply.get("id") != rid
                    or reply.get("sources") != self.expected_sources):
                raise RuntimeError("raw-neural reply identity mismatch")
        except BaseException:
            self.close()
            raise
        for message in reply["warnings"]:
            warnings.warn(message, RuntimeWarning)
        if "error" in reply:
            raise RuntimeError(f"raw neural worker: {reply['error']}\n{reply.get('traceback', '')}")
        return reply["ok"]

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=tabm.SHUTDOWN_SECONDS)
                except (OSError, subprocess.TimeoutExpired):
                    self.proc.kill()
                    self.proc.wait(timeout=tabm.SHUTDOWN_SECONDS)
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream is not None:
                    stream.close()
            self.proc = None


@dataclass
class FittedRawNeural:
    recipe: dict
    features: RawFeaturePipeline = None
    prep: object = None
    states: list = None
    receipt: dict = None
    family: str = "tabm"
    model_spec: dict = None

    def model_inputs(self, raw_rows):
        return self.prep.transform(self.features.transform(raw_rows)).astype("float64")

    def predict_each(self, raw_rows):
        Z = self.prep.transform(self.features.transform(raw_rows))
        outputs = Worker.for_device(self.recipe["adapter"]["device"]).call(
            {"op": "predict", "recipe": self.recipe, "states": self.states, "Z": Z, "model_spec": self.model_spec})
        if (len(outputs) != len(self.recipe["estimator_seeds"])
                or any(p.shape != (len(raw_rows),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in outputs)):
            raise ValueError("invalid raw-neural probabilities")
        return outputs

    def predict_high(self, raw_rows):
        return np.mean(self.predict_each(raw_rows), axis=0)

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, raw_rows):
        high = self.predict_high(raw_rows)
        return np.column_stack([1 - high, high])


def validate_result(recipe, result, rows, y, groups, context, diagnostics):
    _, _, fit_rows, valid, _, _ = backend.validate_request(recipe, rows, y, groups, context, diagnostics)
    receipt = result["receipt"]
    expected_sha = tabm.rows_sha256(rows.index, np.arange(len(rows)))
    if receipt.get("fitting_rows_sha256") != expected_sha or receipt.get("split_seed") != context.split_seed \
            or receipt.get("fold") != context.fold:
        raise ValueError("raw-neural fitted population or context mismatch")
    states, records = result["states"], receipt["tabm"]
    if len(states) != len(recipe["estimator_seeds"]) or len(records) != len(states):
        raise ValueError("raw-neural state/receipt count differs from the seed policy")
    for seed, st in zip(recipe["estimator_seeds"], records):
        expected = {"estimator_seed": seed, "seeds": tabm.stage_seeds(context, seed), "policy": tabm.ITERATIONS,
                    "inner_train_rows_sha256": tabm.rows_sha256(rows.index, fit_rows),
                    "inner_validation_rows_sha256": tabm.rows_sha256(rows.index, valid),
                    "inner_train_rows": len(fit_rows), "inner_validation_rows": len(valid),
                    "stopping_preprocessing_rows_sha256": tabm.rows_sha256(rows.index, fit_rows),
                    "final_preprocessing_rows_sha256": expected_sha,
                    "final_preprocessing_digest": result["prep"].digest_,
                    "encoded_columns": result["prep"].output_columns()}
        if any(st.get(k) != value for k, value in expected.items()):
            raise ValueError("raw-neural stopping receipt contradicts its fitting request")
        p = recipe["params"]
        stopping_history(st, max_epochs=p["max_epochs"], patience=p["patience"], tolerance=p["tolerance"],
                         batch_size=p["batch_size"], fitting_rows=len(rows))
    raw = receipt["raw_neural"]
    if raw["final_model_spec_sha256"] != backend.object_digest(result["model_spec"]):
        raise ValueError("raw-neural embedding specification differs from its fitted state")
    if raw["adapter"] != recipe["adapter"] or raw["diagnostic_assignments"] != (diagnostics is not None):
        raise ValueError("raw-neural receipt declares a different adapter or diagnostic policy")
    if raw["state_sha256"] != [backend.state_digest(s) for s in states]:
        raise ValueError("raw-neural network states differ from their receipt hashes")
    if raw["final_feature_state_sha256"] != backend.object_digest(backend.feature_state(result["features"])):
        raise ValueError("raw-neural final feature state differs from its receipt hash")
    for key, value in raw.items():
        if key.endswith("_sha256") and isinstance(value, str):
            sha256(value, key)


def fit_raw(recipe, rows, y, groups, context, *, diagnostics=None):
    recipe = copy.deepcopy(recipe)
    backend.validate_request(recipe, rows, y, groups, context, diagnostics)
    worker = Worker.for_device(recipe["adapter"]["device"])
    result = worker.call({"op": "fit", "recipe": recipe, "rows": rows, "y": np.asarray(y),
                          "groups": np.asarray(groups), "context": context, "diagnostics": diagnostics})
    validate_result(recipe, result, rows, y, groups, context, diagnostics)
    result["receipt"]["raw_neural"]["worker"] = worker.hello
    return FittedRawNeural(recipe, result["features"], result["prep"], result["states"], result["receipt"],
                           recipe["family"], result["model_spec"])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker" or sys.argv[2] not in ("cpu", "cuda"):
        raise SystemExit("use --worker cpu|cuda")
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    from songpop.phase2 import raw_neural
    try:
        raw_neural.serve(sys.argv[2], sys.stdin.buffer, out)
    finally:
        out.close()
