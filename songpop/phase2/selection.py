"""The frozen selection rule (plan section 11, steps 1 to 3), task P2.6a: a pure function of the candidates'
evidence and the frozen menu. It sees only each candidate's out-of-fold probabilities and fold scores on the
population and folds it is given: no query predictions, no public score, no outer labels, no global bank."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

TIE_BAND = 0.001
TIE_KEYS = ("atomic_recipes", "supervised_transformations", "learned_combiners", "complexity_rank")


@dataclass(frozen=True)
class Evidence:
    """One candidate's evaluation on one population: labels, groups and folds (R, N) of that population, the
    candidate's out-of-fold probabilities (R, N) and fold AUCs (R, K)."""
    candidate_id: str
    y: np.ndarray
    groups: np.ndarray
    fold_id: np.ndarray
    oof: np.ndarray
    fold_auc: np.ndarray


def check_evidence(ev, first, n_repeats, n_folds):
    """Complete on the declared repeats and folds, on the same population and folds as the first candidate, with
    every fold AUC recomputed from the probabilities."""
    R, N = ev.oof.shape
    if (R, ev.fold_auc.shape[1]) != (n_repeats, n_folds) or ev.fold_auc.shape[0] != R or ev.fold_id.shape != (R, N):
        raise ValueError(f"{ev.candidate_id}: evidence is not complete on {n_repeats} repeats of {n_folds} folds")
    if not (np.array_equal(ev.y, first.y) and np.array_equal(ev.groups, first.groups)
            and np.array_equal(ev.fold_id, first.fold_id)):
        raise ValueError(f"{ev.candidate_id}: another population or other folds than {first.candidate_id}")
    if not np.isfinite(ev.oof).all() or ev.oof.min() < 0 or ev.oof.max() > 1:
        raise ValueError(f"{ev.candidate_id}: probabilities missing or outside [0, 1]")
    redo = np.array([[roc_auc_score(ev.y[ev.fold_id[r] == k], ev.oof[r, ev.fold_id[r] == k]) for k in range(n_folds)]
                     for r in range(R)])
    if not np.allclose(redo, ev.fold_auc, rtol=0, atol=1e-12):
        raise ValueError(f"{ev.candidate_id}: fold AUCs differ from its probabilities")


def select_candidate(evidence, menu, n_repeats=3, n_folds=5):
    """Section 11 steps 1 to 3. evidence: {candidate id: Evidence}; menu: {"candidates": [{"id", "atomic_recipes",
    "supervised_transformations", "learned_combiners", "complexity_rank"}, ...]}. Exactly the menu's candidates,
    each complete; ranked by unrounded mean fold AUC; the tie set is every candidate within TIE_BAND of the single
    maximum (not a chained comparison); within it the lexicographic minimum of (atomic recipes, supervised feature
    transformations, learned combiners, complexity rank, candidate id)."""
    cands = {c["id"]: c for c in menu["candidates"]}
    if len(cands) != len(menu["candidates"]) or set(evidence) != set(cands):
        raise ValueError(f"evidence for {sorted(evidence)} but the menu names {sorted(cands)}")
    first = evidence[menu["candidates"][0]["id"]]
    for ev in evidence.values():
        check_evidence(ev, first, n_repeats, n_folds)
    means = {cid: float(ev.fold_auc.mean()) for cid, ev in evidence.items()}
    top = max(means.values())
    tied = sorted(cid for cid, m in means.items() if top - m <= TIE_BAND)
    key = lambda cid: tuple(cands[cid][k] for k in TIE_KEYS) + (cid,)  # noqa: E731
    chosen = min(tied, key=key)
    return {"selected": chosen, "maximum": top, "means": means, "tied": tied,
            "tie_keys": {cid: list(key(cid)) for cid in tied}}
