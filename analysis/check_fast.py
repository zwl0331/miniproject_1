"""Quick checks of the retained validation boundary, not the old P2.9 suite."""
import numpy as np

from analysis.fast_exploration import data_inputs
from analysis.fast_features import FastFeatures, HierarchicalDate
from songpop.phase2.encoders import FitContext, inner_group_splits


def main():
    data, rows, test, groups, folds = data_inputs()
    print("PASS: saved folds, row IDs, labels, groups and test IDs match both reference banks")
    ix = np.flatnonzero(folds[0] != 0)[:400]
    x, y, g = rows.iloc[ix].copy(), data.y[ix].copy(), groups[ix]
    x.loc[:, "track_album_release_date"] = "2020-01-01"
    ctx = FitContext(155, 0)
    splits = inner_group_splits(y, g, ctx.inner_seed("date_te"))
    tr, va = splits[0]
    changed = y.copy()
    changed[va] = 1 - changed[va]
    good = HierarchicalDate().fit_transform(x, y, g, ctx, splits)
    other = HierarchicalDate().fit_transform(x, changed, g, ctx, splits)
    assert np.array_equal(good.iloc[va].to_numpy(), other.iloc[va].to_numpy())
    bad = HierarchicalDate().fit(x, y).transform(x).to_numpy()[va]
    bad_changed = HierarchicalDate().fit(x, changed).transform(x).to_numpy()[va]
    assert not np.array_equal(bad, bad_changed), "known leaky control was not detected"
    assert not np.array_equal(good.iloc[tr].to_numpy(), other.iloc[tr].to_numpy()), "positive label control failed"
    print("PASS: grouped hierarchical encoding excludes held-out labels; deliberately leaky full-fit control fails")
    opts = {"round": True, "round_detail": True, "audio_extra": True, "hier": [30, 10, 3],
            "date_counts": True, "year_relative": True}
    feat = FastFeatures(opts)
    train = feat.fit_transform(rows.iloc[ix], data.y[ix], groups[ix], ctx)
    before = feat.transform(test.iloc[:10])
    poison = test.iloc[:10].copy()
    poison.loc[:, "duration_ms"] = 1e9
    feat.transform(poison)
    after = feat.transform(test.iloc[:10])
    assert before.equals(after)
    assert list(train.columns) == list(before.columns)
    assert not np.isinf(train.to_numpy()).any()
    print("PASS: feature shapes and fitted state survive query poisoning; no test features fitted")


if __name__ == "__main__":
    main()
