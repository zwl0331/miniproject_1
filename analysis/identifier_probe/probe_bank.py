"""Serialization and validation of the P2.8 probe banks (plan section 16.2): phase 1's ten bank arrays, validated by
songpop.evaluate.validate_bank, with probe metadata under the schema p28_probe_bank_v1. A probe bank is diagnostic:
it is never selection-eligible and always names why it is not deployable. Phase 1's save_bank and load_bank serve
bank_v1 only, so probe banks are written and read here."""
import io
import json
from pathlib import Path

import numpy as np

from songpop import artifacts
from songpop.evaluate import N_SPLITS, PredictionBank, validate_bank

SCHEMA = "p28_probe_bank_v1"
REQUIRED = ("task_id", "fit", "spec", "token_source", "selection_eligible", "not_deployable_reason",
            "forest_prediction_threads", "warnings", "completed_fits", "receipts", "config_sha256", "source_manifest")
STABLE = ("schema",) + REQUIRED  # every metadata field but elapsed_seconds, which differs on replay


def validate(bank):
    """The arrays by phase 1's rules, then the probe metadata; raises ValueError."""
    validate_bank(bank, N_SPLITS)
    md = bank.metadata
    if md.get("schema") != SCHEMA:
        raise ValueError(f"probe bank schema {md.get('schema')!r}, expected {SCHEMA!r}")
    missing = [k for k in REQUIRED if k not in md]
    if missing:
        raise ValueError(f"probe bank metadata lacks {missing}")
    if md["task_id"] != "P2.8" or md["selection_eligible"] is not False or not md["not_deployable_reason"]:
        raise ValueError("a probe bank belongs to P2.8, is never selection-eligible and names why it is not deployable")
    if md["completed_fits"] != bank.fold_auc.size or len(md["receipts"]) != bank.fold_auc.size:
        raise ValueError("probe bank fit count or receipts do not match its folds")
    return bank


def to_bytes(bank):
    return artifacts.bank_bytes(validate(bank))


def from_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        arrays = {k: z[k] for k in PredictionBank.ARRAYS}
        meta = json.loads(str(z["metadata_json"]))
    return validate(PredictionBank(**arrays, metadata=meta))


def save(bank, path):
    """Validate, then write once; an existing file is never overwritten. Returns the sha256 of the bytes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"probe bank {path} exists; banks are immutable")
    raw = to_bytes(bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(raw)
    return artifacts.sha256_bytes(raw)
