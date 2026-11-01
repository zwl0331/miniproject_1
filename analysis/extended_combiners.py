"""Existing regularized combiners plus a convex pairwise-ranking weight fit."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from songpop.phase2.procedures import check_meta, clip, fit_combiner, apply_combiner

SETTINGS = ([("constrained", x) for x in (0., .01, .1)] +
            [("stacker", x) for x in (.01, .1, 1.)] +
            [("pairwise", x) for x in (0., .01, .1)])


def fit_pairwise(Z, y, regularization, *, seed=155, max_pairs=50000):
    Z, y = check_meta(Z, y)
    L = logit(clip(Z))
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(positive) * len(negative) <= max_pairs:
        differences = (L[positive, None] - L[None, negative]).reshape(-1, Z.shape[1])
    else:
        rng = np.random.default_rng(seed)
        differences = L[rng.choice(positive, max_pairs)] - L[rng.choice(negative, max_pairs)]
    equal = np.full(Z.shape[1], 1 / Z.shape[1])
    def objective(weights):
        margin = differences @ weights
        loss = np.logaddexp(0., -margin).mean() + regularization * np.square(weights - equal).sum()
        gradient = -(differences.T @ expit(-margin)) / len(margin) + 2 * regularization * (weights - equal)
        return float(loss), gradient
    result = minimize(objective, equal, jac=True, method="SLSQP", bounds=[(0., 1.)] * len(equal),
        constraints=[dict(type="eq", fun=lambda w: w.sum()-1, jac=lambda w: np.ones_like(w))],
        options=dict(ftol=1e-10, maxiter=1000))
    weights = result.x
    if not result.success or not np.isfinite(weights).all() or weights.min() < -1e-8 or abs(weights.sum()-1) > 1e-8:
        raise ValueError(f"pairwise weight optimization failed: {result.message}")
    weights = np.maximum(weights, 0); weights /= weights.sum()
    if objective(weights)[0] > objective(equal)[0] + 1e-9:
        raise ValueError("pairwise fitted weights worsen the starting objective")
    return dict(kind="pairwise", regularization=regularization, weights=weights.tolist(),
                pair_seed=seed, pair_count=len(differences), objective=objective(weights)[0],
                iterations=int(result.nit), clip=1e-6)


def fit(setting, Z, y):
    if setting not in SETTINGS:
        raise ValueError("unlisted combiner setting")
    return fit_pairwise(Z, y, setting[1]) if setting[0] == "pairwise" else fit_combiner(setting, Z, y)


def apply(state, Z):
    if state["kind"] == "pairwise":
        return expit(logit(clip(Z)) @ np.asarray(state["weights"]))
    return apply_combiner(state, Z)
