"""Prediction bank, provenance, experiment log, submissions (doc/plans/phase1-data-baseline.md sections 4.6, 4.8)."""
import hashlib
import io
import json
import os
import platform
import subprocess
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
import xgboost

from songpop.evaluate import BANK_SCHEMA, PredictionBank, validate_bank
from songpop.features import make_features

THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
# Metadata that must be identical when a bank is recomputed; times, the command and the git head may differ.
STABLE_METADATA = ["schema", "run_id", "recipe_id", "task_id", "protocol", "selection_eligible", "model",
                   "feature_set", "feature_definition", "features", "parameters", "data_hashes", "split_version",
                   "split_file", "split_sha256", "grouping", "excluded_row_ids", "split_seeds", "estimator_seed",
                   "deployment", "source_manifest", "config", "config_sha256", "warnings", "completed_fits"]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dumps(obj):
    """Canonical JSON: sorted keys, no NaN or infinity."""
    return json.dumps(obj, sort_keys=True, allow_nan=False)


def source_manifest(root, paths):
    """sha256 of each executing source file and of the sorted manifest of those hashes."""
    files = {str(p): sha256_bytes((Path(root) / p).read_bytes()) for p in sorted(map(str, paths))}
    manifest = "".join(f"{h} {p}\n" for p, h in files.items())
    return {"files": files, "manifest_sha256": sha256_bytes(manifest.encode())}


def packages():
    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}


def config_hash(config):
    return sha256_bytes(dumps(config).encode())


def environment(root):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    pools = [{k: p.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
             for p in threadpoolctl.threadpool_info()]
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages(), "requirements_sha256": sha256_bytes((Path(root) / "requirements.txt").read_bytes()),
            "git_head": head, "threads": {v: os.environ.get(v) for v in THREAD_VARS},
            "threadpools": pools, "cpu_count": os.cpu_count()}


def bank_bytes(bank):
    buf = io.BytesIO()
    np.savez(buf, **{f: getattr(bank, f) for f in PredictionBank.ARRAYS},
             metadata_json=np.array(dumps(bank.metadata)))
    return buf.getvalue()


def save_bank(bank, path):
    """Validate, then write once; an existing file is never overwritten. Returns the sha256 of the bytes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"bank {path} exists; banks are immutable")
    validate_bank(bank, require_metadata=True)
    b = bank_bytes(bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(b)
    return sha256_bytes(b)


def load_bank(source, require_metadata=True):
    """Load from a path or bytes without pickle, and validate."""
    f = io.BytesIO(source) if isinstance(source, bytes) else source
    with np.load(f, allow_pickle=False) as z:
        arrays = {k: z[k] for k in PredictionBank.ARRAYS}
        meta = json.loads(str(z["metadata_json"]))
    bank = PredictionBank(**arrays, metadata=meta)
    validate_bank(bank, require_metadata=require_metadata)
    return bank


def compare_banks(a, b, atol=1e-12):
    """Fields that differ: ids, labels, groups, seeds and folds exactly; predictions and AUCs within atol;
    the stable metadata exactly, reported as metadata:<key>."""
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
    for k in STABLE_METADATA:
        if dumps(a.metadata.get(k)) != dumps(b.metadata.get(k)):
            bad.append(f"metadata:{k}")
    return bad


def append_experiment(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(dumps(record) + "\n")


def read_experiments(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def submission_bytes(test_id, predictions):
    """The CSV bytes of a submission: ID and Popularity_Type, the probability of High at full precision."""
    df = pd.DataFrame({"ID": np.asarray(test_id), "Popularity_Type": np.asarray(predictions, dtype="float64")})
    return df.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()


def write_submission(path, data, bank):
    """Write the bank's deployment output as a submission file, never over an existing file; read the bytes
    back from the destination and validate those. Returns the validation of the read-back bytes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"submission {path} exists; submissions are never overwritten")
    content = submission_bytes(bank.test_id, bank.test_mean)
    validate_submission(content, data, bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)
    with open(path, "rb") as f:
        back = f.read()
    report = validate_submission(back, data, bank)
    if back != content:
        raise ValueError("the bytes read back differ from the bytes written")
    report["path"] = str(path)
    return report


def _feature_keys(X):
    """A hashable key per row of a feature frame; NaN equals NaN."""
    return [tuple("nan" if v != v else v for v in row) for row in X.to_numpy(dtype="float64")]


def validate_submission(content, data, bank, tol=1e-12):
    """Check submission bytes against the sample, the validated bank's deployment output, and the bank's own
    recipe features, derived here from data.test. Raises ValueError on the first failure."""
    validate_bank(bank, require_metadata=True)
    X_test = make_features(data.test, bank.metadata["feature_set"])
    if list(X_test.columns) != list(bank.metadata["features"]):
        raise ValueError("the bank's recorded features differ from its feature set's columns")
    if list(bank.test_id) != list(data.test_id):
        raise ValueError("the bank's test ids differ from the test file's")
    df = pd.read_csv(io.BytesIO(content))
    if list(df.columns) != ["ID", "Popularity_Type"]:
        raise ValueError(f"columns are {list(df.columns)}, expected ['ID', 'Popularity_Type']")
    if len(df) != len(data.sample):
        raise ValueError(f"{len(df)} rows, expected {len(data.sample)}")
    if not df["ID"].is_unique:
        raise ValueError("duplicate IDs")
    if list(df["ID"]) != list(data.sample["ID"]):
        raise ValueError("IDs differ from the sample submission's order")
    p = pd.to_numeric(df["Popularity_Type"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(p)):
        raise ValueError("missing or non-finite probabilities")
    if p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities outside [0, 1]")
    if not np.allclose(p, bank.test_mean, rtol=0, atol=tol):
        raise ValueError(f"probabilities differ from the bank's deployment output by up to "
                         f"{np.max(np.abs(p - bank.test_mean)):.3g}")
    rows_by_key, key_of = {}, _feature_keys(X_test)
    for i, key in enumerate(key_of):
        rows_by_key.setdefault(key, []).append(i)
    positions = data.test.reset_index(drop=True)
    pairs = [list(g.index) for _, g in positions.groupby("uri") if len(g) > 1]
    for idx in pairs:
        if len({key_of[i] for i in idx}) != 1:
            raise ValueError("a repeated uri pair in test has different recipe features")
    spread, n_dup = 0.0, 0
    for idx in rows_by_key.values():
        if len(idx) > 1:
            n_dup += 1
            spread = max(spread, float(np.ptp(p[idx])))
    if spread > tol:
        raise ValueError(f"test rows with identical features have predictions differing by {spread:.3g}")
    q = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    return {"rows": len(df), "sha256": sha256_bytes(content), "bank_run_id": bank.metadata["run_id"],
            "min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()),
            "quantiles": dict(zip(map(str, q), np.quantile(p, q).round(6).tolist())),
            "unique_values": int(len(np.unique(p))), "at_0_or_1": int(((p == 0) | (p == 1)).sum()),
            "identical_feature_groups": n_dup, "max_spread_within_identical": spread,
            "repeated_uri_pairs_checked": len(pairs)}
