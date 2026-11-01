"""bank_v2: the phase 1 arrays with phase 2 metadata (plan sections 4.2, 4.4 and 4.5). Saved once, loaded with an
explicit fold count, validated by the unchanged phase 1 array validator and then by validate_v2_metadata."""
import io
import json
import sys
from pathlib import Path

import numpy as np

from songpop import artifacts
from songpop.evaluate import N_SPLITS, PredictionBank, validate_bank
from songpop.phase2.features import feature_columns
from songpop.phase2.recipes import recipe_hash, recipe_id
from songpop.phase2.receipts import legacy_tabm
from songpop.splits import SPLIT_SEEDS

BANK_SCHEMA_V2 = "bank_v2"
KINDS = ("fitted", "seed_average", "blend")
TOL = 1e-12  # prediction tolerance for classical models (plan section 4.4)
REQUIRED = ["schema", "run_id", "task_id", "stage", "prediction_kind", "selection_eligible", "n_splits",
            "n_repeats", "split_seeds", "evaluation_population", "fitting_population", "features",
            "feature_lineage", "parents", "config", "config_sha256", "source_manifest", "environment",
            "warnings", "completed_fits"]
# Metadata that must be identical when a bank is recomputed; times, the command and the git head may differ.
STABLE = ["schema", "run_id", "recipe", "recipe_id", "recipe_sha256", "task_id", "stage", "prediction_kind",
          "selection_eligible", "n_splits", "n_repeats", "split_seeds", "estimator_seeds", "evaluation_population",
          "fitting_population", "features", "feature_lineage", "fit_receipts", "parents", "config", "config_sha256",
          "source_manifest", "warnings", "completed_fits"]
WEIGHT_TOL = 1e-12


def source_manifest(root):
    """sha256 of every project source file loaded in this process (the phase 2 modules and the top-level
    modules they import) and of the running script, with the hash of the sorted manifest."""
    root = Path(root).resolve()
    paths = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py"):
            p = Path(f).resolve()
            if root in p.parents and ".venv" not in p.parts:
                paths.add(p.relative_to(root))
    main = Path(sys.argv[0]).resolve()
    if main.suffix == ".py" and root in main.parents:
        paths.add(main.relative_to(root))
    return artifacts.source_manifest(root, sorted(map(str, paths)))


def evaluation_population(row_ids):
    ids = np.asarray(row_ids, dtype=np.int64)
    return {"rows": int(len(ids)), "row_ids_sha256": artifacts.sha256_bytes(ids.tobytes())}


def run_identity(config):
    """The identity hash of a run: every field of plan section 4.2 that changes the computation."""
    return artifacts.config_hash(config)


def make_run_id(name, stage, config):
    return f"p2_{name}_{stage}_{run_identity(config)[:12]}"


def canonical_seeds(seeds):
    """Requested split seeds in the official order of songpop.splits.SPLIT_SEEDS; repeats are rows of that order."""
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(SPLIT_SEEDS):
        raise ValueError(f"split seeds {seeds} must be distinct members of {SPLIT_SEEDS}")
    return [s for s in SPLIT_SEEDS if s in seeds]


def stage_of(split_seeds):
    return "seeds" + "-".join(str(int(s)) for s in split_seeds)


def validate_v2_metadata(bank):
    """Keys, consistency with the arrays, the recipe and the identity hashes. Raises ValueError."""
    md = bank.metadata
    missing = [k for k in REQUIRED if k not in md]
    if missing:
        raise ValueError(f"bank_v2 metadata lacks {missing}")
    if md["schema"] != BANK_SCHEMA_V2:
        raise ValueError(f"bank schema {md['schema']!r}, expected {BANK_SCHEMA_V2!r}")
    if md["prediction_kind"] not in KINDS:
        raise ValueError(f"prediction kind {md['prediction_kind']!r} not in {KINDS}")
    R, K = bank.fold_auc.shape
    if md["n_splits"] != K or md["n_repeats"] != R or md["split_seeds"] != [int(s) for s in bank.split_seeds]:
        raise ValueError("metadata fold count, repeat count or split seeds differ from the arrays")
    if md["split_seeds"] != canonical_seeds(md["split_seeds"]) or md["stage"] != stage_of(md["split_seeds"]):
        raise ValueError("split seeds are not distinct production seeds in canonical order, or the stage differs")
    cfg = md["config"]
    links = {"split_seeds": cfg.get("split_seeds", cfg.get("parent_seeds")) == md["split_seeds"],
             "evaluation_population": cfg.get("evaluation_population") == md["evaluation_population"],
             "source_manifest_sha256": cfg.get("source_manifest_sha256") == md["source_manifest"]["manifest_sha256"],
             "source manifest digest": md["source_manifest"]["manifest_sha256"] == artifacts.sha256_bytes(
                 "".join(f"{h} {p}\n" for p, h in sorted(md["source_manifest"]["files"].items())).encode()),
             "parents": cfg.get("parents") == md["parents"]}
    bad = [k for k, ok in links.items() if not ok]
    if bad:
        raise ValueError(f"metadata contradicts its config in {bad}")
    if md["selection_eligible"] != (sorted(md["split_seeds"]) == sorted(SPLIT_SEEDS)):
        raise ValueError("selection_eligible must say exactly whether the bank holds all three split seeds")
    if md["evaluation_population"] != evaluation_population(bank.train_row_id):
        raise ValueError("metadata evaluation population differs from the bank's row ids")
    if md["config_sha256"] != run_identity(md["config"]):
        raise ValueError("config_sha256 is not the hash of config")
    if not md["run_id"].endswith(md["config_sha256"][:12]):
        raise ValueError("run_id does not carry the config hash")
    if md["prediction_kind"] == "blend":
        parents = md["parents"]
        if len(parents) < 2 or not all({"run_id", "bank_sha256", "weight"} <= set(p) for p in parents):
            raise ValueError("a blend needs at least two parents with run id, bank hash and weight")
        w = np.array([p["weight"] for p in parents], dtype="float64")
        if len({p["run_id"] for p in parents}) != len(parents) or not np.all(np.isfinite(w)) or np.any(w <= 0) \
                or abs(w.sum() - 1.0) > WEIGHT_TOL:
            raise ValueError("blend parents must be distinct with finite positive weights summing to one")
        if cfg.get("kind") != "blend" or cfg.get("parent_seeds") != md["split_seeds"]:
            raise ValueError("a blend's config must be of kind blend with its parent seeds")
    else:
        for k in ("recipe", "recipe_id", "recipe_sha256", "estimator_seeds"):
            if k not in md:
                raise ValueError(f"bank_v2 metadata of a fitted bank lacks {k}")
        if md["recipe_sha256"] != recipe_hash(md["recipe"]) or md["recipe_id"] != recipe_id(md["recipe"]):
            raise ValueError("recipe hash or id differs from the stored recipe")
        if md["estimator_seeds"] != md["recipe"]["estimator_seeds"]:
            raise ValueError("estimator seeds differ from the recipe's")
        if md["prediction_kind"] != ("fitted" if len(md["estimator_seeds"]) == 1 else "seed_average"):
            raise ValueError("prediction kind does not match the number of estimator seeds")
        fold_hash = artifacts.sha256_bytes(np.ascontiguousarray(bank.fold_id, dtype=np.int64).tobytes())
        links = {"kind": cfg.get("kind") == "fitted", "recipe_sha256": cfg.get("recipe_sha256") == md["recipe_sha256"],
                 "estimator_seeds": cfg.get("estimator_seeds") == md["estimator_seeds"],
                 "fold_ids_sha256": cfg.get("fold_ids_sha256") == fold_hash,
                 "features": md["features"] == feature_columns(md["recipe"]),
                 "fit_receipts": isinstance(md.get("fit_receipts"), list) and len(md["fit_receipts"]) == R * K}
        bad = [k for k, ok in links.items() if not ok]
        if bad:
            raise ValueError(f"fitted-bank metadata contradicts its config, arrays or recipe in {bad}")
        if md["recipe"]["schema"] == "recipe_v1" and md["recipe"]["family"] == "tabm":
            from songpop.phase2.encoders import FitContext
            for r, seed in enumerate(bank.split_seeds):
                for k in range(K):
                    fitting = bank.fold_id[r] != k
                    legacy_tabm(md["recipe"], md["fit_receipts"][r * K + k], bank.train_row_id[fitting],
                                bank.y[fitting], bank.group_id[fitting], FitContext(int(seed), k))
    if sorted(md["feature_lineage"]) != sorted(md["features"]):
        raise ValueError("feature lineage does not cover exactly the features")


def validate_v2(bank, n_splits=N_SPLITS):
    validate_bank(bank, n_splits=n_splits, require_metadata=False)
    validate_v2_metadata(bank)


def save_bank_v2(bank, path, n_splits=N_SPLITS):
    """Validate, then write once; an existing file is never overwritten. Returns the sha256 of the bytes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"bank {path} exists; banks are immutable")
    validate_v2(bank, n_splits)
    b = artifacts.bank_bytes(bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(b)
    return artifacts.sha256_bytes(b)


def load_bank_v2(source, n_splits=N_SPLITS, validate=True):
    """Load from a path or bytes without pickle; validate with the explicit fold count. validate=False returns
    the envelope unchecked, so verification can compare its sources before applying current semantics."""
    f = io.BytesIO(source) if isinstance(source, bytes) else source
    with np.load(f, allow_pickle=False) as z:
        arrays = {k: z[k] for k in PredictionBank.ARRAYS}
        meta = json.loads(str(z["metadata_json"]))
    bank = PredictionBank(**arrays, metadata=meta)
    if validate:
        validate_v2(bank, n_splits)
    return bank


def compare_v2(a, b, atol=TOL):
    """Fields that differ: ids, labels, groups, seeds and folds exactly; predictions and AUCs within atol; the
    stable metadata exactly, reported as metadata:<key>."""
    bad = []
    for f in PredictionBank.ARRAYS:
        x, y = getattr(a, f), getattr(b, f)
        if x.shape != y.shape:
            bad.append(f)
        elif f in ("oof", "test_by_fold", "test_mean", "fold_auc"):
            if not np.allclose(x, y, rtol=0, atol=atol):
                bad.append(f)
        elif not np.array_equal(x, y):
            bad.append(f)
    for k in STABLE:
        if artifacts.dumps(a.metadata.get(k)) != artifacts.dumps(b.metadata.get(k)):
            bad.append(f"metadata:{k}")
    return bad
