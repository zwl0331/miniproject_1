"""Contribution blends and the allocation rule for exploration (plan section 5.6). Pure functions of logged banks:
nothing here selects a final candidate or a deployment (plan sections 8 and 11)."""
import numpy as np

from songpop.phase2.ensembles import blend_banks

CONTRIBUTION_WEIGHTS = (0.1, 0.25)  # the challenger's weight; H keeps the rest
CLEAR, SMALLER, COMPETITIVE, ENSEMBLE = 0.003, (0.001, 0.005), (0.0005, 0.002), 0.0005
ALLOWANCE_AFTER_PROGRESS = 2


def repeat_means(bank):
    """Mean fold AUC per repeat, unrounded."""
    return np.asarray(bank.fold_auc, dtype="float64").mean(axis=1)


def contribution(h, h_sha, c, c_sha, weight):
    """The declared blend (1 - weight) H + weight c on identical folds: its unrounded mean fold AUC and repeat
    means. Both banks must hold the same rows, labels, groups, split seeds, folds and test ids."""
    b = blend_banks([(h, h_sha), (c, c_sha)], [1.0 - weight, weight])
    return {"weight": float(weight), "mean_fold_auc": float(b.fold_auc.mean()), "repeat_mean_auc": repeat_means(b).tolist()}


def gain(a, b, margin):
    """a beats b by at least margin in unrounded mean, with a positive difference on every repeat."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    return bool(a.mean() - b.mean() >= margin and np.all(a - b > 0))


def within(a, ref, distance):
    """a's unrounded mean is at most distance below ref's."""
    return bool(np.mean(ref) - np.mean(a) <= distance)


def routes(config, incumbent, strongest, h, previous_best_blend):
    """The routes of plan section 5.6 a newly evaluated configuration meets. Each argument is a list of repeat
    means; config also carries "blends", the repeat means of its contribution blends with H."""
    s = config["standalone"]
    met = []
    if gain(s, incumbent, CLEAR):
        met.append("clear_standalone")
    if gain(s, incumbent, SMALLER[0]) and within(s, strongest, SMALLER[1]):
        met.append("smaller_useful")
    if gain(s, incumbent, COMPETITIVE[0]) and within(s, strongest, COMPETITIVE[1]):
        met.append("competitive_refinement")
    if any(gain(b, h, ENSEMBLE) and gain(b, previous_best_blend, ENSEMBLE) for b in config["blends"]):
        met.append("ensemble_contribution")
    return met


def next_allowance(allowance, progress, initial):
    """The family's counter: the initial batch grants two expansion batches on progress against the family anchor
    and stops the family otherwise; a later batch with progress resets it to two, one without uses one."""
    if initial:
        return ALLOWANCE_AFTER_PROGRESS if progress else 0
    return ALLOWANCE_AFTER_PROGRESS if progress else allowance - 1
