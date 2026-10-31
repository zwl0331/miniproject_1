"""Checks of the P2.7 driver's own wiring (Codex rounds W item 7 and X item 10), run before the long stages:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_checks.py
Each check fails on a specific wrong implementation: the path gate at its boundary; a saved assignment that is
reordered, substituted, split across groups, non-integer or single-class; a measurement record with a failed
reproduction, a missing atom, wrong arithmetic or a wrong reduced candidate; a loaded result with a wrong fold id,
an inconsistent selection, a wrong tie set or foreign provenance; a replay comparison that misses a corrupted
prediction, a changed selection, a changed tie set, a drift just above the tolerance or changed engine sources;
a receipt whose files were altered; the freshness count for 0 to 3 learned selections; the atomic cache."""
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import p27_execute as D  # noqa: E402
from songpop.phase2 import procedures as P, runs  # noqa: E402


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")


def main():
    ok = []
    assert D.decide_path(11_520.0, 8)[0] == "full" and D.decide_path(11_519.9, 8)[0] == "full"
    assert D.decide_path(11_520.1, 8)[0] == "reduced" and D.decide_path(1e9, 8)[0] == "reduced"
    ok.append("path gate: full at and below the boundary, reduced above it")
    ctx = runs.Context.load(Path("."))
    pop = P.Population(ctx.rows.iloc[:600], ctx.data.y[:600].copy(), ctx.groups[:600].copy(), "checks")
    fold = D.nested.outer_folds(pop)
    good = pd.read_csv(pd.io.common.StringIO(D.assignment_text(pop, fold)))
    assert np.array_equal(D.check_assignment(good, pop), fold)
    bad = good.copy(); bad.loc[0, "train_row_id"] = bad.loc[1, "train_row_id"]
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    raises(lambda: D.check_assignment(good.iloc[::-1].reset_index(drop=True), pop), ValueError)
    bad = good.copy(); bad["outer_fold"] = bad["outer_fold"].astype(float)
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    g = pop.groups; multi = next(x for x in np.unique(g) if (g == x).sum() > 1)
    bad = good.copy(); pos = np.flatnonzero(g == multi); bad.loc[pos[0], "outer_fold"] = (bad.loc[pos[0], "outer_fold"] + 1) % 3
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    bad = good.copy(); bad.loc[pop.y == 1, "outer_fold"] = 0
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    ok.append("assignment: substituted, reordered, non-integer, split-group and single-class assignments refused")
    # the measurement record against the menu
    menu, _ = D.load_menu()
    meas = json.loads((D.OUT / "measure.json").read_text()) if (D.OUT / "measure.json").exists() else None
    if meas is not None:
        assert D.validate_measure_record(meas, menu) == []
        m = copy.deepcopy(meas); m["atoms_equal_logged_banks"][next(iter(m["atoms_equal_logged_banks"]))] = False
        assert D.validate_measure_record(m, menu)
        m = copy.deepcopy(meas); k = next(iter(m["atom_seconds"])); del m["atom_seconds"][k]
        assert D.validate_measure_record(m, menu)
        m = copy.deepcopy(meas); m["c_all_seconds"] += 1.0
        assert D.validate_measure_record(m, menu)
        m = copy.deepcopy(meas); m["reduced_candidate"] = menu["production_selection"]["selected"]
        assert D.validate_measure_record(m, menu)
        m = copy.deepcopy(meas); m["path"] = "reduced" if m["path"] == "full" else "full"
        assert D.validate_measure_record(m, menu)
        m = copy.deepcopy(meas); m["reserved_hours"] = 1000
        assert D.validate_measure_record(m, menu)
        ok.append("measurement record: a failed reproduction, a missing atom, wrong sums, a wrong reduced candidate, a wrong path and a wrong reservation refused")
    # a synthetic full-path result over a two-candidate menu
    keys = {"x": (1, 0, 0, 0, "x"), "y": (1, 0, 0, 1, "y")}
    stable = {"path": "full", "menu_sha256": "m", "executable_sha256": "e", "split_sha256": "s", "measure_sha256": "q"}
    manifest = {"files": {"songpop/phase2/nested.py": "aaa", D.DRIVER: "bbb"}, "manifest_sha256": "h"}

    def result(shift=0.0, selected="x", means=None, tied=None, fold_ids=(0, 1, 2), prov=None):
        means = means or {"x": 0.84, "y": 0.8395}
        tied = list(tied) if tied is not None else sorted(c for c, m in means.items() if max(means.values()) - m <= D.TIE_BAND)
        folds = []
        for o, fid in zip(range(3), fold_ids):
            v = np.flatnonzero(fold == o)
            folds.append({"fold": fid, "selected": selected, "validation_row_ids": pop.row_ids[v],
                          "selection": {"selected": selected, "maximum": max(means.values()), "means": dict(means),
                                        "tied": tied, "tie_keys": {c: list(keys[c]) for c in tied}},
                          "fold_average": np.full(len(v), 0.5 + shift), "full_refit": np.full(len(v), 0.4),
                          "anchor": np.full(len(v), 0.3)})
        return {"outer_fold": fold, "row_ids": pop.row_ids, "reduced": False, "folds": folds,
                "provenance": dict(prov or {**stable, "source_manifest": manifest})}
    a = result()
    D.validate_result(a, pop, fold, keys, None, stable)
    raises(lambda: D.validate_result(result(fold_ids=(0, 1, 99)), pop, fold, keys, None, stable), ValueError)
    b = result(); b["folds"][1]["selection"]["selected"] = "y"
    raises(lambda: D.validate_result(b, pop, fold, keys, None, stable), ValueError)
    raises(lambda: D.validate_result(result(tied=["x"]), pop, fold, keys, None, stable), ValueError)
    raises(lambda: D.validate_result(result(selected="y"), pop, fold, keys, None, stable), ValueError)  # y loses the tie-break
    raises(lambda: D.validate_result(result(prov={**stable, "menu_sha256": "other", "source_manifest": manifest}), pop, fold, keys, None, stable), ValueError)
    raises(lambda: D.validate_result(result(means={"x": 0.84, "z": 0.8}), pop, fold, keys, None, stable), ValueError)
    ok.append("loaded result: a wrong fold id, a selection that contradicts the selected id, a wrong tie set, a tie-break loser, foreign provenance and a foreign candidate refused")
    assert D.compare_results(a, result())[0]
    assert not D.compare_results(a, result(shift=1e-9))[0]
    assert D.compare_results(a, result(shift=1e-13))[0]
    assert not D.compare_results(a, result(selected="y", means={"x": 0.84, "y": 0.8415}))[0]
    assert not D.compare_results(a, result(tied=["x"]))[0]
    c = result(); c["folds"][1]["selection"]["means"]["y"] += 1e-9
    assert not D.compare_results(a, c)[0]
    assert not D.compare_results(a, result(fold_ids=(0, 2, 1)))[0]
    other_engine = {"files": {"songpop/phase2/nested.py": "changed", D.DRIVER: "ccc"}, "manifest_sha256": "h2"}
    assert not D.compare_results(a, result(prov={**stable, "source_manifest": other_engine}))[0]
    driver_only = {"files": {"songpop/phase2/nested.py": "aaa", D.DRIVER: "ccc"}, "manifest_sha256": "h3"}
    assert D.compare_results(a, result(prov={**stable, "source_manifest": driver_only}))[0]
    ok.append("replay comparison: a 1e-9 drift, a changed selection, a changed tie set, a drifted inner mean, reordered fold ids and changed engine sources fail; 1e-13 and a changed driver alone pass")
    # the receipt binding
    saved_out = D.OUT
    with tempfile.TemporaryDirectory() as tmp:
        D.OUT = Path(tmp)
        raises(lambda: D.require_receipt(stable), RuntimeError)
        for k in ("build", "replay"):
            (Path(tmp) / f"result_{k}.npz").write_bytes(b"npz" + k.encode()); (Path(tmp) / f"result_{k}.json").write_bytes(b"json")
        files = {k: {"npz": "", "json": "", **D.file_hashes(Path(tmp) / f"result_{k}")} for k in ("build", "replay")}
        receipt = {"schema": D.RECEIPT_SCHEMA, "ok": True, **stable, "files": files, "verified_at_utc": "t"}
        (Path(tmp) / "verification.json").write_text(json.dumps(receipt))
        raises(lambda: D.require_receipt(stable), AssertionError)  # hashes match the files but no log record carries them
        (Path(tmp) / "result_build.npz").write_bytes(b"altered")
        raises(lambda: D.require_receipt(stable), RuntimeError)  # altered bytes
        (Path(tmp) / "verification.json").write_text(json.dumps(dict(receipt, ok=False)))
        raises(lambda: D.require_receipt(stable), RuntimeError)
        (Path(tmp) / "verification.json").write_text(json.dumps(dict(receipt, menu_sha256="other")))
        raises(lambda: D.require_receipt(stable), RuntimeError)
    D.OUT = saved_out
    ok.append("receipt: missing, failed, foreign, altered-file and unlogged receipts refused")
    # the freshness count for 0 to 3 learned selections on the real menu
    ex, _ = D.executable(menu)
    learned_id = next(c["id"] for c in ex["candidates"] if c["kind"] == "learned")
    blend_id = next(c["id"] for c in ex["candidates"] if c["kind"] == "blend")
    for L in range(4):
        r = {"reduced": False, "folds": [{"selected": learned_id if o < L else blend_id} for o in range(3)]}
        assert D.expected_fresh_cross_fits(ex, r) == 294 + 6 * L, (L, D.expected_fresh_cross_fits(ex, r))
    assert D.expected_fresh_cross_fits(ex, {"reduced": True, "folds": [{"selected": blend_id}] * 3}) == 0
    assert D.expected_fresh_cross_fits(ex, {"reduced": True, "folds": [{"selected": learned_id}] * 3}) == 18
    ok.append("freshness: 294 + 6 L cold cross-fits for L learned selections; 0 for the reduced fixed blend, 18 for a learned fixed candidate")
    # the atomic cache
    with tempfile.TemporaryDirectory() as tmp:
        ns = Path(tmp) / "ns"
        c = D.AtomicCache(ns)
        req = {"folds_shape": [1, 5], "n_folds": 2, "fitting": {"rows": 5}, "prediction": {"rows": 3}, "tag": "t"}
        oof, pred = np.full((1, 5), 0.5), np.full((1, 2, 3), 0.25)
        assert c.get(req) is None
        c.put(req, oof, pred)
        assert not [p for p in ns.iterdir() if ".tmp-" in p.name]
        again = D.AtomicCache(ns)
        got = again.get(req)
        assert got is not None and np.array_equal(got[0], oof) and again.hits == 1
        (ns / f"{c.key(req)}.npz.tmp-999").write_bytes(b"leftover")
        assert D.AtomicCache(ns).get(req) is not None
        (ns / f"{c.key(req)}.npz").write_bytes(b"garbage")
        raises(lambda: D.AtomicCache(ns).get(req), Exception)
        raises(lambda: c.put(req, oof * 0.9, pred), ValueError)
    ok.append("atomic cache: published by rename, read back by a fresh object, leftover temporaries ignored, a corrupted entry refused, other predictions refused")
    print("\n".join(f"passed: {m}" for m in ok))


if __name__ == "__main__":
    main()
