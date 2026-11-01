"""Structural validation of saved stopping receipts; safe outside Torch workers.

Historical recipe_v1 receipts retain their schema. Recompute their declared
split, seed and stopping policies from the bank's actual fitting population.
This validates provenance without rerunning neural optimization.
"""
import math
import re
from functools import lru_cache

import numpy as np

from songpop.phase2 import tabm
from songpop.phase2.raw_splits import partition, population


@lru_cache(maxsize=128)
def _stopping_split(labels, groups, folds, seed):
    """Memoize only the pure split construction, keyed by its complete inputs.

Immutable tuples prevent a caller from mutating a cached assignment. Receipts,
row hashes and label/group validity are still checked on every validation.
"""
    fit, valid = tabm.inner_split(np.asarray(labels), np.asarray(groups), folds, seed)
    return tuple(map(int, fit)), tuple(map(int, valid))


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def sha256(value, name):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def stopping_history(receipt, *, max_epochs, patience, tolerance, batch_size, fitting_rows):
    """Check the legacy loss/epoch policy and exact optimizer-update accounting."""
    n = positive_integer(receipt.get("epochs_evaluated"), "epochs_evaluated")
    losses = receipt.get("validation_losses")
    if (not isinstance(losses, list) or len(losses) != n or n > max_epochs
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   for v in losses)):
        raise ValueError("validation_losses must be finite nonnegative losses for exactly the evaluated epochs")
    best = positive_integer(receipt.get("best_epoch"), "best_epoch")
    chosen = positive_integer(receipt.get("chosen_epoch"), "chosen_epoch")
    stopped, expected = tabm.stop_rule(losses, tolerance, patience)
    if best != int(np.argmin(losses)) + 1 or chosen != expected:
        raise ValueError("saved best/chosen epoch disagrees with the declared stopping rule")
    if any(tabm.stop_rule(losses[:end], tolerance, patience)[0] for end in range(1, n)):
        raise ValueError("stopping history continues after its declared patience was exhausted")
    if receipt.get("stopped_by") != ("patience" if stopped else "max_epochs") or (not stopped and n != max_epochs):
        raise ValueError("stopped_by disagrees with the stopping history or epoch cap")
    n_train = positive_integer(receipt.get("inner_train_rows"), "inner_train_rows")
    n_valid = positive_integer(receipt.get("inner_validation_rows"), "inner_validation_rows")
    if n_train + n_valid != fitting_rows:
        raise ValueError("stopping partition row counts do not cover the fitting population")
    for key, expected_updates in (("stopping_updates", math.ceil(n_train / batch_size) * n),
                                  ("final_updates", math.ceil(fitting_rows / batch_size) * chosen)):
        if positive_integer(receipt.get(key), key) != expected_updates:
            raise ValueError(f"{key} disagrees with the declared epoch-to-update policy")


def legacy_tabm(recipe, fit_receipt, row_ids, y, groups, context):
    """Validate one fold's recipe_v1 TabM receipts, in declared estimator order."""
    if recipe.get("schema") != "recipe_v1" or recipe.get("family") != "tabm":
        raise ValueError("legacy TabM receipt validator requires a recipe_v1 TabM recipe")
    y, groups = population(y, groups, row_ids)
    if not isinstance(fit_receipt, dict):
        raise ValueError("fold receipt must be an object")
    fitting_sha = tabm.rows_sha256(row_ids, np.arange(len(row_ids)))
    if (fit_receipt.get("fitting_rows_sha256") != fitting_sha
            or fit_receipt.get("split_seed") != context.split_seed or fit_receipt.get("fold") != context.fold):
        raise ValueError("fold receipt fitting population or context is inconsistent")
    records = fit_receipt.get("tabm")
    if not isinstance(records, list) or len(records) != len(recipe["estimator_seeds"]):
        raise ValueError("one TabM stopping receipt is required per estimator seed")
    for seed, receipt in zip(recipe["estimator_seeds"], records):
        if not isinstance(receipt, dict):
            raise ValueError("TabM stopping receipt must be an object")
        p = tabm.TabMClassifier(**recipe["params"], **tabm.LEGACY_FIT_POLICY,
                               scaler=recipe["preprocessing"]["scaler"], random_state=seed).get_params()
        seeds = tabm.stage_seeds(context, seed)
        if receipt.get("policy") != tabm.ITERATIONS or receipt.get("seeds") != seeds:
            raise ValueError("TabM stopping policy or context-derived seed streams differ")
        if positive_integer(receipt.get("estimator_seed"), "estimator_seed", minimum=0) != seed:
            raise ValueError("TabM estimator seed differs from recipe order")
        fit, valid = partition(y, groups, _stopping_split(tuple(y.tolist()), tuple(groups.tolist()),
                                                        p["inner_folds"], seeds["inner_split"]))
        expected = {
            "inner_train_rows_sha256": tabm.rows_sha256(row_ids, fit),
            "inner_validation_rows_sha256": tabm.rows_sha256(row_ids, valid),
            "stopping_preprocessing_rows_sha256": tabm.rows_sha256(row_ids, fit),
            "final_preprocessing_rows_sha256": fitting_sha,
            "inner_train_rows": len(fit), "inner_validation_rows": len(valid),
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("TabM stopping/preprocessing partitions differ from their permitted rows")
        for key in ("stopping_preprocessing_digest", "final_preprocessing_digest"):
            sha256(receipt.get(key), key)
        columns = receipt.get("encoded_columns")
        if (not isinstance(columns, list) or not columns or not all(isinstance(c, str) and c for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError("encoded_columns must be distinct nonempty names")
        versions = receipt.get("versions")
        if (not isinstance(versions, dict) or set(versions) != {"torch", "tabm"}
                or not all(isinstance(v, str) and v for v in versions.values())):
            raise ValueError("TabM receipt must identify its Torch and TabM runtime versions")
        stopping_history(receipt, max_epochs=p["max_epochs"], patience=p["patience"],
                         tolerance=p["tolerance"], batch_size=p["batch_size"], fitting_rows=len(y))
