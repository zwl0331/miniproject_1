"""Probability blends of banks (plan section 4.6; strategy section 5.5). Estimator seeds are averaged inside a
recipe's own bank (songpop.phase2.engine), so a blend weights exact base recipes, never model families."""
import numpy as np
from sklearn.metrics import roc_auc_score

from songpop.evaluate import PredictionBank, validate_bank
from songpop.metrics import check_pairable

WEIGHT_TOL = 1e-12


def select_seeds(bank, seeds):
    """The bank restricted to the given split seeds, in that order: a view for matched comparisons and blends,
    with the deployment mean recomputed over the kept fold models. Metadata is kept, marked with the seeds."""
    have = [int(s) for s in bank.split_seeds]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(have):
        raise ValueError(f"seeds {seeds} must be distinct members of the bank's {have}")
    idx = [have.index(int(s)) for s in seeds]
    out = PredictionBank(bank.train_row_id.copy(), bank.y.copy(), bank.group_id.copy(), bank.split_seeds[idx].copy(),
                         bank.fold_id[idx].copy(), bank.oof[idx].copy(), bank.test_id.copy(),
                         bank.test_by_fold[idx].copy(), bank.test_by_fold[idx].mean(axis=(0, 1)),
                         bank.fold_auc[idx].copy(), dict(bank.metadata, selected_seeds=[int(s) for s in seeds]))
    validate_bank(out, n_splits=bank.fold_auc.shape[1])
    return out


def blend_banks(members, weights):
    """members: list of (bank, bank_sha256); weights: positive, summing to one. Probabilities are averaged per
    repeat and fold in member order; every AUC is recomputed from the blended predictions."""
    if len(members) < 2 or len(members) != len(weights):
        raise ValueError("a blend needs at least two members and one weight per member")
    w = np.asarray(weights, dtype="float64")
    if not np.all(np.isfinite(w)) or np.any(w <= 0) or abs(w.sum() - 1.0) > WEIGHT_TOL:
        raise ValueError(f"blend weights {weights!r} must be positive and sum to one")
    run_ids = [b.metadata["run_id"] for b, _ in members]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("a run appears twice in the blend")
    first = members[0][0]
    for b, _ in members[1:]:
        check_pairable(first, b)
        if not np.array_equal(first.test_id, b.test_id):
            raise ValueError("banks differ in test_id; they cannot be blended")
    oof = sum(wi * b.oof for wi, (b, _) in zip(w, members))
    test_by_fold = sum(wi * b.test_by_fold for wi, (b, _) in zip(w, members))
    R, K = first.fold_auc.shape
    fold_auc = np.array([[roc_auc_score(first.y[first.fold_id[r] == k], oof[r, first.fold_id[r] == k])
                          for k in range(K)] for r in range(R)])
    bank = PredictionBank(first.train_row_id.copy(), first.y.copy(), first.group_id.copy(), first.split_seeds.copy(),
                          first.fold_id.copy(), oof, first.test_id.copy(), test_by_fold, test_by_fold.mean(axis=(0, 1)),
                          fold_auc)
    bank.metadata.update({
        "prediction_kind": "blend", "warnings": [], "completed_fits": 0,
        "parents": [{"run_id": b.metadata["run_id"], "bank_sha256": sha, "weight": float(wi)}
                    for wi, (b, sha) in zip(w, members)]})
    validate_bank(bank, n_splits=K)
    return bank
