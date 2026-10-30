"""The P2.8 identifier and order probes (doc/plans/phase2-models-features.md section 16.2): the structural audit,
the implementation checks, a score-free timing, the declared menu with its comparisons and controls, and
verification by refitting.

Each mode prints its report; only `run` writes, and only new files under artifacts/p2.8/identifier_probe/ (nothing
is overwritten). The menu, thresholds, salts and bootstrap settings are those of
configs/phase2/p2.8/identifier_probe.json, saved before any probe score; its hash is recorded with every result.
Every probe bank is diagnostic: selection_eligible is false and none is a submission candidate.
From the project root, with PYTHONDONTWRITEBYTECODE=1:
    .venv/bin/python -B -I analysis/identifier_probe/probe.py . audit   > analysis/identifier_probe/audit_output.txt
    .venv/bin/python -B -I analysis/identifier_probe/probe.py . checks  > analysis/identifier_probe/checks_output.txt
    .venv/bin/python -B -I analysis/identifier_probe/probe.py . timing  > analysis/identifier_probe/timing_output.txt
    .venv/bin/python -B -I analysis/identifier_probe/probe.py . run     > analysis/identifier_probe/run_output.txt
    .venv/bin/python -B -I analysis/identifier_probe/probe.py . verify  > analysis/identifier_probe/verify_output.txt
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "identifier_probe"))

from scipy import sparse  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import blocks as B  # noqa: E402
import probe_bank as PB  # noqa: E402
from songpop import artifacts  # noqa: E402
from songpop.data import AUDIO, DATE  # noqa: E402
from songpop.evaluate import N_SPLITS, PredictionBank, validate_bank  # noqa: E402
from songpop.features import make_features  # noqa: E402
from songpop.metrics import check_pairable, paired_cluster_bootstrap  # noqa: E402
from songpop.models import make_estimator, predict_high  # noqa: E402
from songpop.phase2 import banks  # noqa: E402
from songpop.phase2.encoders import FitContext  # noqa: E402
from songpop.phase2.models import THREADS, prediction_threads  # noqa: E402
from songpop.phase2.runs import Context  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

CONFIG = "configs/phase2/p2.8/identifier_probe.json"
OUT = Path("artifacts/p2.8/identifier_probe")
PHASE1 = {"H": "artifacts/bank/production_lightgbm_F2_9c37e77d549e.npz",
          "R": "artifacts/bank/production_random_forest_F2_58139012bdde.npz"}
TOL = 1e-12
URL_PREFIX = {"track_href": "https://api.spotify.com/v1/tracks/",
              "analysis_url": "https://api.spotify.com/v1/audio-analysis/"}
IDENTIFIER_TESTS = ("I1", "I2", "I3", "I4", "I5", "I6")
MENU_FITS = ["H", "R", "I1", "I2", "I3", "I4", "L", "I7", "I8"]
NOT_DEPLOYABLE = {"I7": "no_train_test_position_mapping"}


def load_config():
    raw = (ROOT / CONFIG).read_bytes()
    cfg = json.loads(raw)
    if list(cfg["fits"]) != MENU_FITS or [c["id"] for c in cfg["comparisons"]] != [f"I{i}" for i in range(1, 9)]:
        raise ValueError("the declared menu is fits H, R, I1 to I4, L, I7, I8 and comparisons I1 to I8, in order")
    for c in cfg["comparisons"]:
        parts = [c["challenger"]] if "challenger" in c else [m for m, _ in c["blend"]]
        if not set(parts + [c["comparator"]]) <= set(cfg["fits"]) or c["kind"] not in cfg["thresholds"]:
            raise ValueError(f"comparison {c['id']} names an undeclared fit or kind")
    return cfg, artifacts.sha256_bytes(raw)


class Probe:
    """The data, folds and tokens of the probe: the 3,863 eligible rows, the saved primary folds of the three
    split seeds (checked by phase 2's Context), F2 as phase 1 builds it, and the parsed uri tokens."""

    def __init__(self):
        self.ctx = Context.load(ROOT)
        d = self.ctx.data
        self.train, self.test = d.train_eligible, d.test
        self.y, self.row_ids, self.test_id = d.y, d.row_ids, d.test_id
        self.groups = self.ctx.groups
        self.seeds = list(SPLIT_SEEDS)
        self.fold_id = self.ctx.fold_id(self.seeds)
        self.X = make_features(self.train, "F2")
        self.X_test = make_features(self.test, "F2")
        self.tokens = B.tokens(self.train["uri"])
        self.test_tokens = B.tokens(self.test["uri"])

    def tokens_for(self, salt):
        if salt is None:
            return self.tokens, self.test_tokens
        return B.pseudotokens(self.tokens, salt), B.pseudotokens(self.test_tokens, salt)


class FittedProbe:
    """One spec fitted on one outer fitting population: its blocks (every dictionary, inner split and encoder
    built from the fitting rows alone) and its estimator. predict_high encodes and scores any other rows."""

    def __init__(self, spec):
        self.spec = spec

    def fit(self, probe, tr, toks, ctx, y=None):
        """tr: positions of the fitting rows; toks: the tokens of every training row (real or pseudotokens); only
        the labels y[tr] are read (y defaults to the probe's labels)."""
        y = probe.y if y is None else y
        self.receipts = []
        if self.spec["model"] == "identifier_linear":
            X = B.hashed_identifier_features(toks[tr])
        else:
            X = probe.X.iloc[tr]
            if self.spec["blocks"]:
                self.blocks = B.FittedBlocks(self.spec["blocks"])
                b = self.blocks.fit_transform(probe.train.iloc[tr], toks[tr], probe.row_ids[tr], y[tr],
                                              probe.groups[tr], ctx)
                b.index = X.index
                X = pd.concat([X, b], axis=1)
                self.receipts = self.blocks.receipts_
        self.columns = None if self.spec["model"] == "identifier_linear" else list(X.columns)
        self.estimator = self._make()
        self.estimator.fit(X, y[tr])
        return self

    def _make(self):
        if self.spec["model"] in ("lightgbm", "random_forest"):
            return make_estimator(self.spec["model"])
        if self.spec["model"] == "identifier_linear":
            return LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=0)
        raise ValueError(f"unknown probe model {self.spec['model']!r}")

    def inputs(self, probe, raw, base, toks, row_ids, positions_known):
        if self.spec["model"] == "identifier_linear":
            return B.hashed_identifier_features(toks)
        if not self.spec["blocks"]:
            return base
        b = self.blocks.transform(raw, toks, row_ids, positions_known)
        b.index = base.index
        X = pd.concat([base, b], axis=1)
        if list(X.columns) != self.columns:
            raise AssertionError("query columns differ from the fitted ones")
        return X

    def predict_high(self, X):
        """Probability of High; a forest predicts with one thread through part C's helper (plan section 16.4). L's
        sparse input gets phase 1's checks of predict_high on its row count (len() is undefined for sparse)."""
        if sparse.issparse(X):
            if list(self.estimator.classes_) != [0, 1]:
                raise ValueError(f"estimator classes are {list(self.estimator.classes_)}, expected 0 and 1")
            proba = np.asarray(self.estimator.predict_proba(X), dtype="float64")
            if proba.shape != (X.shape[0], 2) or not np.isfinite(proba).all() or proba.min() < 0 or proba.max() > 1 \
                    or np.abs(proba.sum(axis=1) - 1).max() > 1e-6:
                raise ValueError("probability matrix fails phase 1's checks")
            return proba[:, 1]
        with prediction_threads(self.spec["model"], [self.estimator]):
            return predict_high(self.estimator, X)


def fit_probe(probe, spec, tr, toks, ctx, y=None):
    return FittedProbe(spec).fit(probe, tr, toks, ctx, y)


def validation_inputs(probe, fitted, va, toks):
    return fitted.inputs(probe, probe.train.iloc[va], probe.X.iloc[va], toks[va], probe.row_ids[va], True)


def test_inputs(probe, fitted, test_toks):
    return fitted.inputs(probe, probe.test, probe.X_test, test_toks, probe.X_test.index, False)


def coverage(fitted, Xva, groups_va):
    """Held-out rows and groups the fitted dictionaries support: fragments seen in the fitting population and
    audio keys with support from another group."""
    out = {}
    for c in [c for c in Xva.columns if c.startswith("id_freq_")]:
        out[c] = float((Xva[c].to_numpy() > 0).mean())
    if "audio_support" in Xva.columns:
        s = Xva["audio_support"].to_numpy() > 0
        out["audio_supported_rows"] = int(s.sum())
        out["audio_supported_groups"] = int(len(set(np.asarray(groups_va)[s].tolist())))
    return out


def run_fit(probe, name, spec, salt=None):
    """Cross-validate one declared fit into a bank over the saved primary folds of the three split seeds."""
    toks, test_toks = probe.tokens_for(salt)
    R, N = probe.fold_id.shape
    oof = np.full((R, N), np.nan)
    test_by_fold = np.full((R, N_SPLITS, len(probe.test)), np.nan)
    fold_auc = np.full((R, N_SPLITS), np.nan)
    count = np.zeros((R, N), dtype=int)
    caught, receipts, t0 = [], [], time.time()
    for r, seed in enumerate(probe.seeds):
        for k in range(N_SPLITS):
            va, tr = np.flatnonzero(probe.fold_id[r] == k), np.flatnonzero(probe.fold_id[r] != k)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fitted = fit_probe(probe, spec, tr, toks, FitContext(seed, k))
                Xva = validation_inputs(probe, fitted, va, toks)
                p = fitted.predict_high(Xva)
                test_by_fold[r, k] = fitted.predict_high(test_inputs(probe, fitted, test_toks))
            caught += [f"{type(x.message).__name__}: {x.message}" for x in w]
            cov = coverage(fitted, Xva, probe.groups[va]) if isinstance(Xva, pd.DataFrame) else {}
            receipts.append({"split_seed": seed, "fold": k, "inner": fitted.receipts, "coverage": cov})
            oof[r, va] = p
            count[r, va] += 1
            fold_auc[r, k] = roc_auc_score(probe.y[va], p)
    if not (count == 1).all():
        raise AssertionError("a row was not held out exactly once per repeat")
    bank = PredictionBank(probe.row_ids, probe.y, probe.groups, np.asarray(probe.seeds), probe.fold_id, oof,
                          probe.test_id, test_by_fold, test_by_fold.mean(axis=(0, 1)), fold_auc)
    bank.metadata.update({"schema": "p28_probe_bank_v1", "task_id": "P2.8", "fit": name, "spec": spec,
                          "token_source": "uri" if salt is None else f"pseudotoken:{salt}",
                          "selection_eligible": False,
                          "not_deployable_reason": NOT_DEPLOYABLE.get(name, "diagnostic_probe"),
                          "forest_prediction_threads": 1 if spec["model"] == "random_forest" else None,
                          "warnings": sorted(set(caught)), "completed_fits": R * N_SPLITS, "receipts": receipts,
                          "elapsed_seconds": round(time.time() - t0, 3)})
    validate_bank(bank, N_SPLITS)
    return bank


def blend(a, b, wa):
    """wa x a + (1 - wa) x b on the same rows and folds, fold AUCs recomputed."""
    check_pairable(a, b)
    oof = wa * a.oof + (1 - wa) * b.oof
    tbf = wa * a.test_by_fold + (1 - wa) * b.test_by_fold
    auc = np.array([[roc_auc_score(a.y[a.fold_id[r] == k], oof[r, a.fold_id[r] == k]) for k in range(N_SPLITS)]
                    for r in range(oof.shape[0])])
    return PredictionBank(a.train_row_id, a.y, a.group_id, a.split_seeds, a.fold_id, oof, a.test_id, tbf,
                          tbf.mean(axis=(0, 1)), auc)


def comparison(cfg, cid, challenger, comparator, kind):
    """The declared statistics of challenger minus comparator: unrounded means, repeat differences, the shared
    group bootstrap (95% interval and one-sided multiplicity-adjusted bounds, valid and invalid draws) and the
    follow-up rule of the comparison's kind."""
    bs = cfg["bootstrap"]
    res = paired_cluster_bootstrap(challenger, comparator, n_boot=bs["draws"], seed=bs["seed"], return_draws=True)
    draws = res.pop("draws")
    alpha = bs["alpha"] / bs["family_size"]
    rep = (challenger.fold_auc.mean(axis=1) - comparator.fold_auc.mean(axis=1)).tolist()
    diff = float(challenger.fold_auc.mean() - comparator.fold_auc.mean())
    rule = cfg["thresholds"][kind]
    return {"id": cid, "kind": kind, "challenger_mean_fold_auc": float(challenger.fold_auc.mean()),
            "comparator_mean_fold_auc": float(comparator.fold_auc.mean()), "difference": diff,
            "repeat_differences": rep, "bootstrap": res,
            "lower_bound_one_sided": float(np.quantile(draws, alpha)),
            "upper_bound_one_sided": float(np.quantile(draws, 1 - alpha)),
            "bounds_note": "approximate multiplicity-adjusted conditional bootstrap bounds",
            "follow_up_threshold": rule, "meets_follow_up": bool(diff >= rule and all(x > 0 for x in rep))}


def check_phase1(name, bank):
    """H must equal the phase 1 LightGBM F2 bank in all ten arrays; R (one-thread prediction) its random forest
    bank in every array but the predictions and fold AUCs, its predictions within TOL."""
    ref = artifacts.load_bank(ROOT / PHASE1[name])
    detail = {}
    for f in PredictionBank.ARRAYS:
        x, y = getattr(bank, f), getattr(ref, f)
        detail[f] = float(np.max(np.abs(x - y))) if f in ("oof", "test_by_fold", "test_mean", "fold_auc") \
            else bool(np.array_equal(x, y))
    exact = all(v is True for v in detail.values() if isinstance(v, bool))
    preds = max(detail[f] for f in ("oof", "test_by_fold", "test_mean"))
    ok = exact and (preds == 0.0 and detail["fold_auc"] == 0.0 if name == "H" else preds <= TOL)
    return ok, detail


def evaluate(cfg, fits):
    for name, bank in fits.items():
        check_pairable(fits["H"], bank)  # one global pairability check of every menu bank
    out = []
    for c in cfg["comparisons"]:
        if "blend" in c:
            (m1, w1), (m2, _) = c["blend"]
            chal = blend(fits[m1], fits[m2], w1)
        else:
            chal = fits[c["challenger"]]
        out.append(comparison(cfg, c["id"], chal, fits[c["comparator"]], c["kind"]))
    return out


def pipeline_of(c):
    return c["blend"][1][0] if "blend" in c else c["challenger"]


def control_plan(cfg, menu):
    """Section 16.2: the identifier tests meeting their follow-up rule, ranked by mean gain divided by their
    threshold, ties by test id, and the first max_pipelines distinct fitted pipelines among them (I5 and I6 share
    L, so they share its control fits)."""
    spec_of = {c["id"]: c for c in cfg["comparisons"]}
    cands = [m for m in menu if m["id"] in IDENTIFIER_TESTS and m["meets_follow_up"]]
    cands.sort(key=lambda m: (-m["difference"] / m["follow_up_threshold"], m["id"]))
    pipes = []
    for m in cands:
        p = pipeline_of(spec_of[m["id"]])
        if p not in pipes and len(pipes) < cfg["controls"]["max_pipelines"]:
            pipes.append(p)
    return cands, pipes


def control_stats(cfg, cands, pipes, fits, refits):
    """Each candidate against its comparator with the pseudotoken refits in place of its pipeline; a real gain is
    not clearly distinguished from model perturbation when it is at most the largest control gain plus the margin."""
    spec_of = {c["id"]: c for c in cfg["comparisons"]}
    out = []
    for m in cands:
        c = spec_of[m["id"]]
        p = pipeline_of(c)
        if p not in pipes:
            out.append({"id": m["id"], "pipeline": p, "controlled": False, "reason": "beyond the pipeline limit"})
            continue
        rows = []
        for s in cfg["controls"]["salts"]:
            if "blend" in c:
                (m1, w1), _ = c["blend"]
                ctrl = blend(fits[m1], refits[(p, s)], w1)
            else:
                ctrl = refits[(p, s)]
            rows.append({"salt": s, **comparison(cfg, m["id"], ctrl, fits[c["comparator"]], c["kind"])})
        largest = max(r["difference"] for r in rows)
        out.append({"id": m["id"], "pipeline": p, "controlled": True, "real_difference": m["difference"],
                    "controls": rows, "largest_control_difference": largest,
                    "distinguished_from_perturbation": bool(m["difference"] > largest + cfg["controls"]["margin"])})
    return out


def controls(probe, cfg, menu, fits):
    cands, pipes = control_plan(cfg, menu)
    refits = {(p, s): run_fit(probe, p, cfg["fits"][p], s) for p in pipes for s in cfg["controls"]["salts"]}
    return control_stats(cfg, cands, pipes, fits, refits), refits


def summary_line(m):
    b = m["bootstrap"]
    if m.get("role") == "descriptive only":
        return (f"  {m['id']} (descriptive only): {m['challenger_mean_fold_auc']:.6f} vs "
                f"{m['comparator_mean_fold_auc']:.6f}, difference {m['difference']:+.6f}, by repeat "
                f"{', '.join(f'{x:+.6f}' for x in m['repeat_differences'])}; 95% [{b['interval_95'][0]:+.6f}, "
                f"{b['interval_95'][1]:+.6f}], draws {b['valid']} valid, {b['invalid']} invalid")
    return (f"  {m['id']} ({m['kind']}): {m['challenger_mean_fold_auc']:.6f} vs {m['comparator_mean_fold_auc']:.6f}, "
            f"difference {m['difference']:+.6f}, by repeat {', '.join(f'{x:+.6f}' for x in m['repeat_differences'])}; "
            f"95% [{b['interval_95'][0]:+.6f}, {b['interval_95'][1]:+.6f}], one-sided bounds "
            f"[{m['lower_bound_one_sided']:+.6f}, {m['upper_bound_one_sided']:+.6f}], draws {b['valid']} valid, "
            f"{b['invalid']} invalid; follow-up {'met' if m['meets_follow_up'] else 'not met'} "
            f"(threshold {m['follow_up_threshold']})")


def coverage_summary(fits):
    out = {}
    for name in ("I2", "I3", "I8"):
        covs = [r["coverage"] for r in fits[name].metadata["receipts"]]
        out[name] = {k: [c[k] for c in covs] for k in covs[0]}
    return out


def descriptive(cfg, fits):
    """I3 minus I2, the target encodings' increment over the fragment counts: described only, never a follow-up
    route and never a control candidate (section 16.2)."""
    d = comparison(cfg, "I3-I2", fits["I3"], fits["I2"], "block")
    for k in ("kind", "follow_up_threshold", "meets_follow_up", "lower_bound_one_sided", "upper_bound_one_sided",
              "bounds_note"):
        d.pop(k)  # no follow-up rule and no place in the menu's multiplicity family
    d["role"] = "descriptive only"
    return [d]


def assemble_results(cfg, cfg_sha, manifest, fits, ctrl, saved):
    """The results file: everything a verification recomputes, apart from the bank hashes it checks first."""
    return {"schema": "p28_probe_results_v1", "config": CONFIG, "config_sha256": cfg_sha, "source_manifest": manifest,
            "banks": saved, "menu": evaluate(cfg, fits), "descriptive": descriptive(cfg, fits), "controls": ctrl,
            "coverage": coverage_summary(fits), "L_mean_fold_auc": float(fits["L"].fold_auc.mean())}


def check_results(cfg, cfg_sha, saved, stored, refitted, manifest_now):
    """Every difference between the saved results and banks and their refits under the code running now: the
    configuration and source provenance, the set of banks, each bank's arrays and stable metadata, and every
    statistic recomputed from the refitted banks (menu, descriptive comparison, controls, coverage, L's score).
    refitted holds new banks by name, with config_sha256 and source_manifest set by the caller. Returns a list."""
    problems = []
    if saved["config_sha256"] != cfg_sha:
        problems.append("the declared menu changed since the results were written")
    if saved["source_manifest"] != manifest_now:
        problems.append("the recorded source manifest differs from the code running now")
    fits = {n: refitted[n] for n in cfg["fits"] if n in refitted}
    if len(fits) != len(cfg["fits"]):
        return problems + ["a menu bank is missing"]
    menu = evaluate(cfg, fits)
    cands, pipes = control_plan(cfg, menu)
    expected = list(cfg["fits"]) + [f"{p}__{x}" for p in pipes for x in cfg["controls"]["salts"]]
    if sorted(saved["banks"]) != sorted(expected) or sorted(stored) != sorted(expected) \
            or not set(expected) <= set(refitted):
        return problems + [f"saved banks {sorted(saved['banks'])} differ from the expected {sorted(expected)}"]
    for name in expected:
        old, new = stored[name], refitted[name]
        bad = [f for f in PredictionBank.ARRAYS if not np.array_equal(getattr(new, f), getattr(old, f))]
        bad += [f"metadata:{k}" for k in PB.STABLE
                if artifacts.dumps(new.metadata.get(k)) != artifacts.dumps(old.metadata.get(k))]
        if bad:
            problems.append(f"{name} differs from its refit in {bad}")
    refits = {(p, x): refitted[f"{p}__{x}"] for p in pipes for x in cfg["controls"]["salts"]}
    again = assemble_results(cfg, cfg_sha, manifest_now, fits, control_stats(cfg, cands, pipes, fits, refits),
                             saved["banks"])
    for key in ("menu", "descriptive", "controls", "coverage", "L_mean_fold_auc"):
        if artifacts.dumps(again[key]) != artifacts.dumps(saved.get(key)):
            problems.append(f"{key} recomputed from the refits differs from the saved results")
    return problems


def mode_run():
    cfg, cfg_sha = load_config()
    if (ROOT / OUT).exists():
        raise FileExistsError(f"{OUT} exists; the probe is run once")
    probe = Probe()
    fits = {name: run_fit(probe, name, spec) for name, spec in cfg["fits"].items()}
    for name in ("H", "R"):
        ok, detail = check_phase1(name, fits[name])
        print(f"{name} against its phase 1 bank: {'OK' if ok else 'FAILED'} {detail}")
        if not ok:
            sys.exit(1)
    menu = evaluate(cfg, fits)
    for m in menu + descriptive(cfg, fits):
        print(summary_line(m))
    print(f"  L alone: mean fold AUC {fits['L'].fold_auc.mean():.6f} (diagnostic)")
    for name, c in coverage_summary(fits).items():
        print(f"  coverage {name}: " + "; ".join(f"{k} {min(v):.4g} to {max(v):.4g}" for k, v in c.items()))
    ctrl, refits = controls(probe, cfg, menu, fits)
    for c in ctrl:
        if not c["controlled"]:
            print(f"  control {c['id']}: not run ({c['reason']})")
            continue
        print(f"  control {c['id']} ({c['pipeline']}): real {c['real_difference']:+.6f}, largest pseudotoken "
              f"{c['largest_control_difference']:+.6f}: "
              f"{'distinguished' if c['distinguished_from_perturbation'] else 'not clearly distinguished'} "
              f"from model perturbation")
    if not ctrl:
        print("  controls: no identifier test met its follow-up rule; none run")
    manifest = banks.source_manifest(ROOT)
    saved = {}
    for name, bank in list(fits.items()) + [(f"{p}__{x}", b) for (p, x), b in refits.items()]:
        bank.metadata.update({"config_sha256": cfg_sha, "source_manifest": manifest})
        saved[name] = PB.save(bank, ROOT / OUT / "banks" / f"{name}.npz")
    results = assemble_results(cfg, cfg_sha, manifest, fits, ctrl, saved)
    (ROOT / OUT / "results.json").write_text(artifacts.dumps(results) + "\n")
    print(f"saved {len(saved)} banks and {OUT / 'results.json'}")


def mode_verify():
    """Refit every saved bank under the code running now and check everything check_results covers; the recorded
    source manifest must also equal its files on disk. Any problem exits 1."""
    cfg, cfg_sha = load_config()
    saved = json.loads((ROOT / OUT / "results.json").read_text())
    probe = Probe()
    problems, stored, refitted = [], {}, {}
    for name, sha in saved["banks"].items():
        raw = (ROOT / OUT / "banks" / f"{name}.npz").read_bytes()
        if artifacts.sha256_bytes(raw) != sha:
            problems.append(f"{name}: bank differs from its recorded hash")
            continue
        stored[name] = PB.from_bytes(raw)
        fit, _, salt = name.partition("__")
        refitted[name] = run_fit(probe, fit, cfg["fits"][fit], salt or None)
        print(f"{name}: refitted")
    rec = saved["source_manifest"]
    if artifacts.source_manifest(ROOT, list(rec["files"])) != rec:
        problems.append("a source file of the recorded manifest changed on disk")
    manifest_now = banks.source_manifest(ROOT)
    for bank in refitted.values():
        bank.metadata.update({"config_sha256": cfg_sha, "source_manifest": manifest_now})
    problems += check_results(cfg, cfg_sha, saved, stored, refitted, manifest_now)
    for p in problems:
        print("  PROBLEM:", p)
    print("verification:", "PASSED" if not problems else "FAILED")
    sys.exit(0 if not problems else 1)


def mode_timing():
    """Score-free: the seconds to fit each declared fit on split seed 155, fold 0, and predict its rows, and to
    draw 1,000 shared bootstrap replicates of H against itself; no AUC of any probe fit is computed, nothing kept."""
    cfg, _ = load_config()
    probe = Probe()
    tr, va = np.flatnonzero(probe.fold_id[0] != 0), np.flatnonzero(probe.fold_id[0] == 0)
    toks, test_toks = probe.tokens_for(None)
    total = 0.0
    for name, spec in cfg["fits"].items():
        t0 = time.time()
        fitted = fit_probe(probe, spec, tr, toks, FitContext(probe.seeds[0], 0))
        fitted.predict_high(validation_inputs(probe, fitted, va, toks))
        fitted.predict_high(test_inputs(probe, fitted, test_toks))
        dt = time.time() - t0
        total += dt * 15
        print(f"  {name}: {dt:.2f} s for one fold, about {dt * 15:.0f} s for its 15 fits")
    ref = artifacts.load_bank(ROOT / PHASE1["H"])
    t0 = time.time()
    paired_cluster_bootstrap(ref, ref, n_boot=1000, seed=cfg["bootstrap"]["seed"])
    boot = (time.time() - t0) * cfg["bootstrap"]["draws"] / 1000
    n_ctrl = cfg["controls"]["max_pipelines"] * len(cfg["controls"]["salts"])
    print(f"menu: about {total / 60:.1f} minutes of fitting; each comparison's {cfg['bootstrap']['draws']} draws "
          f"about {boot:.0f} s, {len(cfg['comparisons'])} comparisons about {boot * len(cfg['comparisons']) / 60:.1f} "
          f"minutes; controls at most {n_ctrl} further 15-fit evaluations; a projection, not a bound")


def mode_audit():
    """The structural audit of section 16.2: no labels, no fitting."""
    probe = Probe()
    d = probe.ctx.data
    pops = {"eligible training rows": d.train_eligible, "all training rows": d.train, "test rows": d.test}
    for label, f in pops.items():
        u = f["uri"]
        ok = u.dropna().map(lambda s: bool(B.URI.fullmatch(s)))
        print(f"{label}: {len(f)} rows; uri missing {int(u.isna().sum())}; well-formed tokens {int(ok.sum())}")
        toks = B.tokens(u.dropna())
        print(f"  first characters: {''.join(sorted(set(t[0] for t in toks)))}")
        cover = [len(set(t[i] for t in toks)) for i in range(1, B.TOKEN_LENGTH)]
        print(f"  distinct characters at positions 1 to 21: min {min(cover)}, max {max(cover)} of {len(B.ALPHABET)}")
        for col, pre in URL_PREFIX.items():
            sub = f[[col, "uri"]].dropna()
            exact = (sub[col] == pre + sub["uri"].str[len("spotify:track:"):]).sum()
            print(f"  {col}: missing {int(f[col].isna().sum())}; equal to {pre} + token: {int(exact)} of {len(sub)}")
        print(f"  type values: {f['type'].value_counts(dropna=False).to_dict()}")
        print(f"  uri multiplicity (copies: number of uris): {u.value_counts().value_counts().sort_index().to_dict()}")
    tr, te = set(probe.tokens), set(probe.test_tokens)
    print(f"full-token overlap between eligible training and test: {len(tr & te)}")
    for n in ("p1", "p2", "p3", "s1", "s2", "s3"):
        a = set(B.fragments(probe.tokens, [n])[n])
        b = B.fragments(probe.test_tokens, [n])[n]
        print(f"  test rows whose {n} occurs in training: {b.isin(a).mean():.4f} ({len(a)} distinct in training)")
    ids = d.test["ID"].to_numpy()
    gaps = np.diff(ids)
    print(f"test ID: unique {len(set(ids)) == len(ids)}, increasing {bool((gaps > 0).all())}, range {ids.min()} to "
          f"{ids.max()}, gaps {pd.Series(gaps).value_counts().sort_index().to_dict()}")
    ka = B.audio_keys(d.train_eligible)["audio"].to_numpy()
    kt = B.audio_keys(d.test)["audio"]
    kda = set(ka + "#" + d.train_eligible[DATE].to_numpy())
    kdt = kt + "#" + d.test[DATE].to_numpy()
    hit = kt.isin(set(ka))
    print(f"exact audio-plus-date overlap, test rows: {int(kdt.isin(kda).sum())}; audio-only: {int(hit.sum())} rows, "
          f"{kt[hit].nunique()} distinct signatures")
    support = B.GroupCount({"audio": "s"}).fit(pd.DataFrame({"audio": ka}), probe.groups).count_["audio"]
    multi = sum(1 for v in support.values() if v > 1)
    print(f"audio keys of eligible training rows held by more than one primary group: {multi} of {len(support)}")
    year = pd.Series(d.train_eligible[DATE].str[:4].astype(int).to_numpy())
    test_year = pd.Series(d.test[DATE].str[:4].astype(int).to_numpy())
    print(f"training row position against release year: Spearman "
          f"{pd.Series(probe.row_ids).corr(year, method='spearman'):+.4f}; test ID against release year: Spearman "
          f"{pd.Series(ids).corr(test_year, method='spearman'):+.4f}")
    gc = B.GroupCount({"t": "whole_token"}).fit(pd.DataFrame({"t": probe.tokens}), probe.groups)
    own = np.array([gc.count_["t"][t] for t in probe.tokens])
    print(f"whole-token distinct groups per token on eligible rows: max {own.max()} (a fitting row's own group "
          f"left out gives {int((own - 1).max())}); on test rows, with the full training dictionary: "
          f"{int(sum(gc.count_['t'].get(t, 0) for t in probe.test_tokens))} in total")
    print(f"tempo: minimum {d.train_eligible['tempo'].min()} (eligible rows), {d.test['tempo'].min()} (test rows)")


def mode_checks():
    """Implementation checks of section 16.2, on the real rows and hand-computed cases, without any probe score."""
    cfg, _ = load_config()
    probe = Probe()
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'}  {name}")

    tr, va = np.flatnonzero(probe.fold_id[0] != 0), np.flatnonzero(probe.fold_id[0] == 0)
    ctx = FitContext(probe.seeds[0], 0)
    toks, test_toks = probe.tokens_for(None)
    allb = {"model": "lightgbm", "blocks": ["id_shape_v1", "id_frequency_v1", "id_te30_v1", "row_position",
                                             "audio_recurrence"]}

    def inputs(y=None, tt=test_toks):
        f = fit_probe(probe, allb, tr, toks, ctx, y)
        fb = B.FittedBlocks(allb["blocks"])
        yy = probe.y if y is None else y
        fit_rows = fb.fit_transform(probe.train.iloc[tr], toks[tr], probe.row_ids[tr], yy[tr], probe.groups[tr], ctx)
        return fit_rows, validation_inputs(probe, f, va, toks), test_inputs(probe, f, tt)

    base = inputs()
    y2 = probe.y.copy()
    y2[va] = 1 - y2[va]
    flip = inputs(y=y2)
    check("held-out labels flipped (outer and inner splits fixed) change nothing fitted",
          all(a.equals(b) for a, b in zip(base, flip)))
    alt = inputs(tt=B.pseudotokens(test_toks, "test-isolation"))
    check("other test tokens change no fitting or validation input", base[0].equals(alt[0]) and base[1].equals(alt[1]))
    perm = np.random.default_rng(0).permutation(len(tr))
    names = ["id_shape_v1", "id_frequency_v1", "audio_recurrence"]
    fb1, fb2 = B.FittedBlocks(names), B.FittedBlocks(names)
    a = fb1.fit_transform(probe.train.iloc[tr], toks[tr], probe.row_ids[tr], probe.y[tr], probe.groups[tr], ctx)
    t = tr[perm]
    b = fb2.fit_transform(probe.train.iloc[t], toks[t], probe.row_ids[t], probe.y[t], probe.groups[t], ctx)
    cols = [c for c in a.columns if c != "audio_te_dev"]
    check("row order: row-local and count blocks of fitting rows follow their row ids",
          a[cols].sort_index().equals(b[cols].sort_index()))
    check("row order: held-out transforms are order invariant",
          fb1.transform(probe.train.iloc[va], toks[va], probe.row_ids[va], True).equals(
              fb2.transform(probe.train.iloc[va], toks[va], probe.row_ids[va], True)))
    fb = B.FittedBlocks(["id_frequency_v1", "id_te30_v1", "audio_recurrence"])
    fb.fit_transform(probe.train.iloc[tr], toks[tr], probe.row_ids[tr], probe.y[tr], probe.groups[tr], ctx)
    odd = probe.train.iloc[[0]].copy()
    odd["tempo"] = 1e6  # an audio key no fitting row holds
    z = fb.transform(odd, np.array(["~" * 22], dtype=object), [10 ** 9], True)
    check("an unseen token gets zero counts and the fitting prior; an unsupported audio key zero support and deviation",
          (z.filter(like="id_freq_").to_numpy() == 0).all()
          and np.allclose(z.filter(like="id_te30_").to_numpy(), probe.y[tr].mean(), rtol=0, atol=1e-12)
          and z["audio_support"].iloc[0] == 0 and z["audio_te_dev"].iloc[0] == 0)
    # Counts by hand: group 1 holds two uris with the same prefix "A" (counted once), group 2 a third.
    hand = np.array(["AB" + "0" * 20, "AC" + "0" * 20, "AB" + "1" * 20, "ZZ" + "2" * 20], dtype=object)
    gc = B.GroupCount({"p1": "n1", "p2": "n2"})
    fit = gc.fit_transform(B.fragments(hand, ["p1", "p2"]), [1, 1, 2, 3], pd.Index(range(4)))
    other = gc.transform(B.fragments(np.array(["AQ" + "0" * 20, "Q" * 22], dtype=object), ["p1", "p2"]),
                         pd.Index(range(2)))
    check("group counts by hand: distinct (fragment, group) pairs, own group left out, unseen zero",
          np.allclose(fit["n1"], np.log1p([1, 1, 1, 0])) and np.allclose(fit["n2"], np.log1p([1, 0, 1, 0]))
          and np.allclose(other["n1"], np.log1p([2, 0])) and np.allclose(other["n2"], np.log1p([0, 0])))
    # Target encoding and audio deviation by hand, with explicit inner splits and mixed-label groups.
    yk = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    gk = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    kk = np.array(["a", "a", "b", "b", "a", "b", "a", "b"], dtype=object)
    splits = [(np.array([2, 3, 4, 5, 6, 7]), np.array([0, 1])), (np.array([0, 1, 4, 5, 6, 7]), np.array([2, 3])),
              (np.array([0, 1, 2, 3, 6, 7]), np.array([4, 5])), (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7]))]
    te = B.CrossFittedTE({"k": "te"}, "hand", m=2)
    fitted_te = te.fit_transform(pd.DataFrame({"k": kk}), yk, gk, None, pd.Index(range(8)), splits=splits)["te"]
    dev = B.AudioDeviation(m=2)
    fitted_dev = dev.fit_transform(kk, yk, gk, None, pd.Index(range(8)), splits=splits)["audio_te_dev"].to_numpy()
    want_te, want_dev = np.zeros(8), np.zeros(8)
    for trn, tst in splits:
        mu = yk[trn].mean()
        for i in tst:
            m = kk[trn] == kk[i]
            S, n = yk[trn][m].sum(), m.sum()
            want_te[i] = (S + 2 * mu) / (n + 2)
            want_dev[i] = (S - n * mu) / (n + 2)
    full = te.transform(pd.DataFrame({"k": ["a", "b", "z"]}), pd.Index(range(3)))["te"].to_numpy()
    p = yk.mean()
    want_full = [(yk[kk == c].sum() + 2 * p) / ((kk == c).sum() + 2) for c in ("a", "b")] + [p]
    other_dev = dev.transform(np.array(["a", "z"], dtype=object), pd.Index(range(2)))["audio_te_dev"].to_numpy()
    check("target encoding by hand: cross-fitted with each inner part's prior, other rows by the full population",
          np.allclose(fitted_te, want_te, rtol=0, atol=1e-12) and np.allclose(full, want_full, rtol=0, atol=1e-12))
    check("audio deviation by hand: (S - n mu) / (n + m) in each inner part, zero when unsupported",
          np.allclose(fitted_dev, want_dev, rtol=0, atol=1e-12)
          and np.allclose(other_dev, [(yk[kk == "a"].sum() - (kk == "a").sum() * p) / ((kk == "a").sum() + 2), 0.0]))
    p1, p1b, p2 = (B.pseudotokens(toks, s) for s in ("x", "x", "y"))
    eq_real = pd.Series(toks).groupby(toks).ngroup().to_numpy()
    eq_ps = pd.Series(p1).groupby(p1).ngroup().to_numpy()
    check("pseudotokens: deterministic, format kept, equality preserved, salts differ",
          (p1 == p1b).all() and (p1 != p2).all() and all(B.URI.fullmatch("spotify:track:" + t) for t in p1)
          and pd.Series(toks).duplicated().any() and len(set(zip(eq_real, eq_ps))) == len(set(eq_real)))
    gc = B.GroupCount({"t": "whole"}).fit(pd.DataFrame({"t": toks[tr]}), probe.groups[tr])
    check("whole-token identity: zero support on held-out and test rows",
          all(gc.count_["t"].get(t, 0) == 0 for t in list(toks[va]) + list(test_toks)))
    f = fit_probe(probe, cfg["fits"]["R"], tr, toks, ctx)
    Xte = test_inputs(probe, f, test_toks)
    outs = {f.predict_high(Xte).tobytes() for _ in range(10)}
    check("random forest: ten identical predictions with one thread, four threads restored",
          len(outs) == 1 and f.estimator.n_jobs == THREADS)
    H = fit_probe(probe, cfg["fits"]["H"], tr, toks, ctx)
    check("H's inputs are F2 exactly: the 20 phase 1 columns, no identifier column or placeholder",
          H.columns == list(probe.X.columns) and len(H.columns) == len(AUDIO) + 7)
    # Probe banks: a serialization round trip, its refusals, and the verifier against independent corruptions, on
    # synthetic banks built from the phase 1 bank (perturbed predictions, not probe fits) with 20 bootstrap draws.
    ref = artifacts.load_bank(ROOT / PHASE1["H"])
    n_rows = len(ref.y)

    def fake(name, k, boost=0.0):
        wave = 0.02 * np.sin(np.arange(n_rows) * (k + 1))[None, :]
        oof = np.clip(ref.oof + wave + boost * (ref.y - 0.5)[None, :], 0, 1)
        auc = np.array([[roc_auc_score(ref.y[ref.fold_id[r] == f], oof[r, ref.fold_id[r] == f])
                         for f in range(N_SPLITS)] for r in range(oof.shape[0])])
        bank = PredictionBank(ref.train_row_id, ref.y, ref.group_id, ref.split_seeds, ref.fold_id, oof, ref.test_id,
                              ref.test_by_fold, ref.test_mean, auc)
        cov = {"id_freq_p1": 1.0, "audio_supported_rows": 2, "audio_supported_groups": 1}
        bank.metadata.update({"schema": PB.SCHEMA, "task_id": "P2.8", "fit": name,
                              "spec": cfg["fits"][name.partition("__")[0]], "token_source": "uri",
                              "selection_eligible": False, "not_deployable_reason": "diagnostic_probe",
                              "forest_prediction_threads": None, "warnings": [], "completed_fits": 15,
                              "receipts": [{"split_seed": int(x), "fold": f, "inner": [], "coverage": dict(cov)}
                                           for x in ref.split_seeds for f in range(N_SPLITS)],
                              "config_sha256": "cfg",
                              "source_manifest": {"files": {"a.py": "0"}, "manifest_sha256": "m"}})
        return bank

    copy_of = lambda bank: PB.from_bytes(PB.to_bytes(bank))
    one = fake("I1", 0)
    back = copy_of(one)
    refused = []
    for change in ({"selection_eligible": True}, {"schema": "bank_v1"}, {"not_deployable_reason": ""}):
        bad = copy_of(one)
        bad.metadata.update(change)
        try:
            PB.to_bytes(bad)
        except ValueError:
            refused.append(change)
    check("probe banks: byte round trip keeps arrays and metadata; selection-eligible, bank_v1 and unnamed refused",
          all(np.array_equal(getattr(back, f), getattr(one, f)) for f in PredictionBank.ARRAYS)
          and artifacts.dumps(back.metadata) == artifacts.dumps(one.metadata) and len(refused) == 3)
    cfg_f = json.loads(json.dumps(cfg))
    cfg_f["bootstrap"]["draws"] = 20
    man = {"files": {"a.py": "0"}, "manifest_sha256": "m"}
    fits = {n: fake(n, i, 0.05 if n == "I1" else 0.0) for i, n in enumerate(MENU_FITS)}  # I1 triggers a control
    menu = evaluate(cfg_f, fits)
    cands, pipes = control_plan(cfg_f, menu)
    refits = {(q, x): fake(f"{q}__{x}", 50 + j) for j, (q, x) in
              enumerate((q, x) for q in pipes for x in cfg_f["controls"]["salts"])}
    stored = {**fits, **{f"{q}__{x}": v for (q, x), v in refits.items()}}
    saved = assemble_results(cfg_f, "cfg", man, fits, control_stats(cfg_f, cands, pipes, fits, refits),
                             {n: "sha" for n in stored})
    clean = check_results(cfg_f, "cfg", saved, stored, {n: copy_of(v) for n, v in stored.items()}, man)
    caught = []
    for label in ("L score", "source manifest", "coverage", "arrays", "metadata", "control bank"):
        sv, st = json.loads(json.dumps(saved)), {n: copy_of(v) for n, v in stored.items()}
        rf = {n: copy_of(v) for n, v in stored.items()}
        if label == "L score":
            sv["L_mean_fold_auc"] = -123.0
        elif label == "source manifest":
            sv["source_manifest"] = {"files": {}, "manifest_sha256": "0" * 64}
        elif label == "coverage":
            for r in st["I8"].metadata["receipts"]:
                r["coverage"]["audio_supported_rows"] = 999
            sv["coverage"]["I8"]["audio_supported_rows"] = [999] * 15
        elif label == "arrays":
            rf["I2"].oof[0, 0] = 1 - rf["I2"].oof[0, 0]
        elif label == "metadata":
            st["I4"].metadata["token_source"] = "pseudotoken:x"
        else:
            st.pop(f"I1__{cfg_f['controls']['salts'][0]}", None)
        caught.append(bool(check_results(cfg_f, "cfg", sv, st, rf, man)))
    check(f"the verifier: clean fixture passes ({len(pipes)} control pipeline); a corrupted L score, source manifest, "
          f"coverage, array, metadata field or missing control bank each fails",
          not clean and pipes == ["I1"] and all(caught))
    check("the declared menu: 9 fits, I1 to I8, at most 2 control pipelines, 3 salts, margin 0.0005, 10,000 draws",
          list(cfg["fits"]) == MENU_FITS and cfg["controls"]["max_pipelines"] == 2
          and len(cfg["controls"]["salts"]) == 3 and cfg["controls"]["margin"] == 0.0005
          and cfg["bootstrap"]["draws"] == 10000 and cfg["bootstrap"]["family_size"] == 8)
    print(f"{sum(results)} passed, {len(results) - sum(results)} failed")
    sys.exit(0 if all(results) else 1)


MODES = {"audit": mode_audit, "checks": mode_checks, "timing": mode_timing, "run": mode_run, "verify": mode_verify}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in MODES:
        raise SystemExit(f"usage: probe.py <root> {{{'|'.join(MODES)}}}")
    MODES[sys.argv[2]]()
