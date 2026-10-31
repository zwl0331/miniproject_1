"""The restricted nested evaluation (plan section 8), built in P2.6a and run by P2.7.

Levels: outer, three grouped folds of the eligible rows (split seed 20261101); inner, five grouped folds with split
seeds 155, 2026 and 1029 rebuilt inside each outer training part; for a learned candidate, its combiner-training
folds rebuilt inside each inner training part. In each outer fold every candidate's inner evidence is built from
the outer training rows alone; select_candidate sees only that evidence; the selected candidate's inner-fold average
predicts the outer validation rows; the selected procedure is separately refitted on the whole outer training part;
the anchor's inner-fold average is kept. Outer labels are read only by decide(), after all three folds."""
import numpy as np
from sklearn.metrics import roc_auc_score

from songpop.phase2 import procedures as P
from songpop.phase2.encoders import FitContext
from songpop.phase2.engine import fit_recipe
from songpop.phase2.selection import Evidence, select_candidate

OUTER_SEED = 20261101
OUTER_FOLDS = 3
INNER_SEEDS = (155, 2026, 1029)
INNER_FOLDS = 5
FALLBACK = 0.003  # the anchor is recommended if the selection loses at least this much and on all three folds
REFIT_GAIN = 0.003  # a full refit replaces fold averaging only with at least this gain and on all three folds


def channels_of(cand):
    """The channels of a candidate: an atomic recipe is one channel of one atom."""
    if cand["kind"] == "atomic":
        return [{"id": cand["id"], "atoms": [cand["recipe"]]}]
    return [m["channel"] for m in cand["members"]] if cand["kind"] == "blend" else cand["channels"]


def atoms_of(cand):
    return [a for ch in channels_of(cand) for a in ch["atoms"]]


def candidate_cv(cand, pop, fold_id, seeds, pred_rows, cache, manifest_sha256, packages_of, tag):
    """A candidate cross-fitted inside a population: (out-of-fold (R, N), per-fold predictions (R, K, T))."""
    names = [f"{tag}:{pop.name}:{s}" for s in seeds]
    outer = {ch["id"]: P.channel_cv(ch, pop, fold_id, names, pred_rows, cache, manifest_sha256, packages_of)
             for ch in channels_of(cand)}
    if cand["kind"] == "atomic":
        return outer[cand["id"]]
    if cand["kind"] == "blend":
        w = [m["weight"] for m in cand["members"]]
        return (sum(wi * outer[m["channel"]["id"]][0] for wi, m in zip(w, cand["members"])),
                sum(wi * outer[m["channel"]["id"]][1] for wi, m in zip(w, cand["members"])))
    setting = tuple(cand["setting"])
    got, _ = P.evaluate_learned(cand["channels"], [setting], outer, pop, fold_id, seeds, cache, manifest_sha256,
                                packages_of, protocol=f"{P.META_PROTOCOL}:{tag}")
    return got[setting]["oof"], got[setting]["test"]


def refit_predict(cand, pop, rows, cache, manifest_sha256, packages_of, tag):
    """The complete candidate refitted on all of pop, predicting rows: atoms fitted on pop; a learned candidate also
    regenerates its meta folds and combiner on pop (never reusing weights fitted elsewhere)."""
    def atom(r):
        return fit_recipe(r, pop.rows, pop.y, pop.groups, FitContext(f"{tag}:{pop.name}", "all")).predict_high(rows)

    chans = {ch["id"]: np.mean([atom(a) for a in ch["atoms"]], axis=0) for ch in channels_of(cand)}
    if cand["kind"] == "atomic":
        return chans[cand["id"]]
    if cand["kind"] == "blend":
        return sum(m["weight"] * chans[m["channel"]["id"]] for m in cand["members"])
    setting = tuple(cand["setting"])
    states, _ = P.learn_on(pop, cand["channels"], [setting], P.coordinate_seed(P.META_PROTOCOL, tag, pop.name),
                           f"{P.META_PROTOCOL}:{tag}:{pop.name}", cache, manifest_sha256, packages_of)
    return P.apply_combiner(states[setting], np.column_stack([chans[ch["id"]] for ch in cand["channels"]]))


def outer_folds(pop):
    return P.population_folds(pop.y, pop.groups, pop.row_ids, OUTER_FOLDS, OUTER_SEED)


def inner_folds(O):
    return np.vstack([P.population_folds(O.y, O.groups, O.row_ids, INNER_FOLDS, s) for s in INNER_SEEDS])


def fold_evidence(menu, O, V_rows, cache, manifest_sha256, packages_of, o):
    """Every menu candidate's inner evidence on the outer training part O, and its inner-fold-average predictions of
    the outer validation rows, kept apart from the evidence."""
    F = inner_folds(O)
    evidence, query = {}, {}
    for cand in menu["candidates"]:
        oof, pred = candidate_cv(cand, O, F, INNER_SEEDS, V_rows, cache, manifest_sha256, packages_of, f"nested:{o}")
        auc = np.array([[roc_auc_score(O.y[F[r] == k], oof[r, F[r] == k]) for k in range(INNER_FOLDS)]
                        for r in range(len(INNER_SEEDS))])
        evidence[cand["id"]] = Evidence(cand["id"], O.y, O.groups, F, oof, auc)
        query[cand["id"]] = pred.mean(axis=(0, 1))
    return evidence, query


def run(menu, pop, cache, manifest_sha256, packages_of, anchor_id, reduced=None, fold=None):
    """The nested evaluation of the frozen menu on pop (the eligible rows). Returns per outer fold the selected
    candidate, its fold-average and full-refit predictions of the outer validation rows, and the anchor's; outer
    labels are not read here. reduced: a candidate id fixed by ordinary production CV (the declared reduced path):
    then nothing is selected per fold, the fixed candidate and the anchor are each refitted in the three outer
    training parts, and no decision can follow. fold: the saved outer assignment (default: outer_folds(pop))."""
    fold = outer_folds(pop) if fold is None else np.asarray(fold)
    P.check_folds(pop.y, pop.groups, fold, OUTER_FOLDS)
    by_id = {c["id"]: c for c in menu["candidates"]}
    out = []
    for o in range(OUTER_FOLDS):
        O, V = pop.subset(np.flatnonzero(fold != o), f"outer{o}"), pop.subset(np.flatnonzero(fold == o), f"valid{o}")
        if reduced is None:
            evidence, query = fold_evidence(menu, O, V.rows, cache, manifest_sha256, packages_of, o)
            sel = select_candidate(evidence, menu)
            chosen = sel["selected"]
            rec = {"fold": o, "selected": chosen, "selection": sel, "fold_average": query[chosen],
                   "anchor": query[anchor_id]}
        else:
            chosen, rec = reduced, {"fold": o, "selected": reduced, "selection": None,
                                    "anchor": refit_predict(by_id[anchor_id], O, V.rows, cache, manifest_sha256,
                                                            packages_of, f"nested-full:{o}")}
        rec["full_refit"] = refit_predict(by_id[chosen], O, V.rows, cache, manifest_sha256, packages_of,
                                          f"nested-full:{o}")
        rec["validation_row_ids"] = V.row_ids
        out.append(rec)
    return {"outer_fold": fold, "folds": out, "reduced": reduced is not None}


def decide(result, pop):
    """After all three outer folds (plan section 8): the outer fold AUCs; then the fallback (the selection loses at
    least FALLBACK mean outer AUC to the anchor and loses on all three folds: the anchor with its original
    deployment); only if it does not trigger, the deployment rule (a full refit only with a mean gain of at least
    REFIT_GAIN and a gain on all three folds). The reduced path reports and decides nothing."""
    y = dict(zip(pop.row_ids.tolist(), pop.y.tolist()))
    auc = lambda rec, key: roc_auc_score([y[i] for i in rec["validation_row_ids"]], rec[key])  # noqa: E731
    if result["reduced"]:
        return {"path": "reduced", "full_refit": [auc(r, "full_refit") for r in result["folds"]],
                "anchor": [auc(r, "anchor") for r in result["folds"]], "decision": None}
    fa = np.array([auc(r, "fold_average") for r in result["folds"]])
    an = np.array([auc(r, "anchor") for r in result["folds"]])
    fr = np.array([auc(r, "full_refit") for r in result["folds"]])
    d_anchor, d_refit = fa - an, fr - fa
    if d_anchor.mean() <= -FALLBACK and (d_anchor < 0).all():
        decision = "fallback: the anchor with fold averaging"
    elif d_refit.mean() >= REFIT_GAIN and (d_refit > 0).all():
        decision = "the selected procedure with a full refit"
    else:
        decision = "the selected procedure with fold averaging"
    return {"path": "full", "fold_average": fa.tolist(), "anchor": an.tolist(), "full_refit": fr.tolist(),
            "selected_minus_anchor": d_anchor.tolist(), "refit_minus_fold_average": d_refit.tolist(),
            "selected": [r["selected"] for r in result["folds"]], "decision": decision}
