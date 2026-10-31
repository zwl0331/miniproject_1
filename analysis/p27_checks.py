"""Checks of the P2.7 driver's own wiring (Codex round W, item 7), run before the long stages:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_checks.py
Each check fails on a specific wrong implementation: the path gate at its boundary; a saved assignment that is
reordered, substituted, split across groups or non-integer; a replay comparison that misses a corrupted prediction,
a changed selection, a changed tie set or a drift just above the tolerance; the decide gate without a verified
replay; the atomic cache's publication and its refusal of a corrupted entry."""
import os
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
    # the path gate at its boundary: 28,800 s = 2 * 1.25 * plan projection -> plan projection 11,520 s
    assert D.decide_path(11_520.0, 8)[0] == "full" and D.decide_path(11_519.9, 8)[0] == "full"
    assert D.decide_path(11_520.1, 8)[0] == "reduced" and D.decide_path(1e9, 8)[0] == "reduced"
    ok.append("path gate: full at and below the boundary, reduced above it")
    # the saved assignment
    ctx = runs.Context.load(Path("."))
    pop = P.Population(ctx.rows.iloc[:600], ctx.data.y[:600].copy(), ctx.groups[:600].copy(), "checks")
    fold = D.nested.outer_folds(pop)
    good = pd.read_csv(pd.io.common.StringIO(D.assignment_text(pop, fold)))
    assert np.array_equal(D.check_assignment(good, pop), fold)
    bad = good.copy(); bad.loc[0, "train_row_id"] = bad.loc[1, "train_row_id"]
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    bad = good.iloc[::-1].reset_index(drop=True)
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    bad = good.copy(); bad["outer_fold"] = bad["outer_fold"].astype(float)
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    g = pop.groups; multi = next(x for x in np.unique(g) if (g == x).sum() > 1)  # a repeated group
    bad = good.copy(); pos = np.flatnonzero(g == multi)
    bad.loc[pos[0], "outer_fold"] = (bad.loc[pos[0], "outer_fold"] + 1) % 3
    raises(lambda: D.check_assignment(bad, pop), ValueError)
    bad = good.copy(); bad["outer_fold"] = np.where(pop.y == 1, 0, bad["outer_fold"])  # fold 1 and 2 lose a class? make fold 0 all positives in one fold
    bad2 = good.copy(); bad2.loc[pop.y == 1, "outer_fold"] = 0
    raises(lambda: D.check_assignment(bad2, pop), ValueError)
    ok.append("assignment: substituted, reordered, non-integer, split-group and single-class assignments refused")
    # the replay comparison on a synthetic result
    rng = np.random.default_rng(0)
    def result(seed_shift=0.0, selected="x", tied=("x", "y")):
        folds = []
        for o in range(3):
            v = np.flatnonzero(fold == o)
            folds.append({"fold": o, "selected": selected, "validation_row_ids": pop.row_ids[v],
                          "selection": {"selected": selected, "maximum": 0.84, "means": {"x": 0.84, "y": 0.8395},
                                        "tied": list(tied), "tie_keys": {c: [1, 0, 0, 0, c] for c in tied}},
                          "fold_average": rng.random(len(v)) * 0 + 0.5 + seed_shift, "full_refit": np.full(len(v), 0.4),
                          "anchor": np.full(len(v), 0.3)})
        return {"outer_fold": fold, "row_ids": pop.row_ids, "reduced": False, "folds": folds}
    a = result()
    assert D.compare_results(a, result())[0]
    assert not D.compare_results(a, result(seed_shift=1e-9))[0]
    assert D.compare_results(a, result(seed_shift=1e-13))[0]
    assert not D.compare_results(a, result(selected="y"))[0]
    assert not D.compare_results(a, result(tied=("x",)))[0]
    b = result(); b["folds"][1]["selection"]["means"]["y"] += 1e-9
    assert not D.compare_results(a, b)[0]
    ok.append("replay comparison: a 1e-9 drift, a changed selection, a changed tie set and a drifted inner mean fail; 1e-13 passes")
    # the decide gate
    saved_out = D.OUT
    with tempfile.TemporaryDirectory() as tmp:
        D.OUT = Path(tmp)
        raises(D.require_replay_verified, RuntimeError)
        (Path(tmp) / "replay.txt").write_text("replay: DIFFERENCES FOUND\n")
        raises(D.require_replay_verified, RuntimeError)
        (Path(tmp) / "replay.txt").write_text("x\nreplay: every array within 1e-12, every selection equal, every cross-fit fresh\n")
        D.require_replay_verified()
    D.OUT = saved_out
    ok.append("decide gate: refused without a replay record or with a failed one")
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
        (ns / f"{c.key(req)}.npz.tmp-999").write_bytes(b"leftover")  # a leftover temporary file is ignored
        assert D.AtomicCache(ns).get(req) is not None
        path = ns / f"{c.key(req)}.npz"
        path.write_bytes(b"garbage")  # a corrupted entry is refused, not a miss
        raises(lambda: D.AtomicCache(ns).get(req), Exception)
        raises(lambda: c.put(req, oof * 0.9, pred), ValueError)
    ok.append("atomic cache: published by rename, read back by a fresh object, leftover temporaries ignored, a corrupted entry refused, other predictions refused")
    print("\n".join(f"passed: {m}" for m in ok))


if __name__ == "__main__":
    main()
