"""P2.9 boundary checks, including known-wrong leakage controls.

Run from the root with .venv/bin/python -m scripts.check_p29_boundaries.
No model fitting or scientific scoring. Legacy implementation stays unchanged.
"""
import copy
import json
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from songpop.data import AUDIO, DATE
from songpop.phase2.encoders import FitContext
from songpop.phase2.features import FeaturePipeline, RAW_COLUMNS
from songpop.phase2.raw_features import RawFeaturePipeline
from songpop.phase2.raw_splits import crossfit, partition, population


def fixture():
    groups = np.repeat(np.arange(20), 2)
    y = (groups % 2).astype(np.int64)
    rows = pd.DataFrame({c: np.linspace(.1, .9, len(y)) for c in AUDIO})
    rows["key"], rows["mode"], rows["time_signature"] = groups % 12, groups % 2, 4
    rows["duration_ms"], rows["tempo"], rows["loudness"] = 180000., 120., -10.
    rows[DATE] = [f"{2000 + int(g) % 4}-01-01" for g in groups]
    rows.index = 100 + np.arange(len(y)) * 3
    splits = [(np.flatnonzero(groups % 5 != k), np.flatnonzero(groups % 5 == k)) for k in range(5)]
    recipe = {"base": "F2", "blocks": ["date_count", "date_te30", "era_relative", "music"], "drop": []}
    return rows[RAW_COLUMNS], y, groups, splits, recipe


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.rows, self.y, self.groups, self.splits, self.recipe = fixture()

    def test_population_rejects_misalignment_missing_and_repeated_identity(self):
        population(self.y, self.groups, self.rows.index)
        bad = [([0] * len(self.y), self.groups, self.rows.index),
               (self.y[:, None], self.groups, self.rows.index),
               (self.y, self.groups[:-1], self.rows.index),
               (self.y, [None] * len(self.y), self.rows.index),
               (self.y, self.groups.astype(float), self.rows.index),
               (self.y, self.groups, [1] * len(self.y)),
               (self.y, self.groups, self.rows.index.to_numpy().astype(float))]
        for args in bad:
            with self.subTest(args_type=str(type(args[0]))), self.assertRaises(ValueError):
                population(*args)

    def test_partition_rejects_leaks_omissions_repetition_and_coercion(self):
        fit, query = self.splits[0]
        valid = partition(self.y, self.groups, (fit[::-1], query))
        np.testing.assert_array_equal(valid[0], fit[::-1])
        self.assertFalse(np.shares_memory(valid[1], query))
        # A complete row partition which splits a primary group is still invalid.
        leaky_fit, leaky_query = fit.copy(), query.copy()
        leaky_fit[0], leaky_query[0] = query[0], fit[0]
        bad = [(fit[:-1], query), (np.append(fit, query[0]), query),
               (np.append(fit[:-1], fit[0]), query), (fit.astype(float), query),
               (fit, query.astype(bool)), (leaky_fit, leaky_query),
               (fit, np.append(query[:-1], len(self.y))), (fit, [])]
        for split in bad:
            with self.subTest(split=str(split)), self.assertRaises(ValueError):
                partition(self.y, self.groups, split)

    def test_crossfit_requires_complements_and_exactly_one_holdout(self):
        crossfit(self.y, self.groups, self.splits)
        bad = [self.splits[:-1], self.splits + [self.splits[0]],
               [self.splits[0]] * 5,
               [(fit[:-2], query) for fit, query in self.splits]]
        for splits in bad:
            with self.subTest(n=len(splits)), self.assertRaises(ValueError):
                crossfit(self.y, self.groups, splits)
        # Single-class query folds are allowed for encoders, not for stopping.
        query = np.flatnonzero(self.groups == 0)
        fit = np.flatnonzero(self.groups != 0)
        partition(self.y, self.groups, (fit, query), query_needs_both_classes=False)
        with self.assertRaises(ValueError):
            partition(self.y, self.groups, (fit, query))

    def test_legacy_features_and_receipts_unchanged_without_injection(self):
        ctx = FitContext(155, 0)
        old, new = FeaturePipeline(self.recipe), RawFeaturePipeline(self.recipe)
        a = old.fit_transform(self.rows, self.y, self.groups, ctx)
        b = new.fit_transform(self.rows, self.y, self.groups, ctx)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        pd.testing.assert_frame_equal(old.transform(self.rows), new.transform(self.rows), check_exact=True)
        self.assertEqual(old.receipts_, new.receipts_)
        self.assertEqual(old.lineage_, new.lineage_)

    def test_heldout_label_isolation_and_known_wrong_encoder(self):
        ctx = FitContext(155, 0)
        changed = self.y.copy()
        fit, query = self.splits[0]
        changed[query] = 1
        a, b = RawFeaturePipeline(self.recipe), RawFeaturePipeline(self.recipe)
        xa = a.fit_transform(self.rows, self.y, self.groups, ctx, encoder_splits=self.splits)
        xb = b.fit_transform(self.rows, changed, self.groups, ctx, encoder_splits=self.splits)
        np.testing.assert_array_equal(xa.iloc[query].date_te30, xb.iloc[query].date_te30)
        self.assertFalse(np.array_equal(xa.iloc[fit].date_te30, xb.iloc[fit].date_te30))
        self.assertEqual(a.receipts_, b.receipts_)
        # This intentionally wrong encoder fits all labels, including query groups.
        # The very same isolation assertion must reject it (test sensitivity).
        def wrong(labels):
            f = pd.DataFrame({"date": self.rows[DATE].to_numpy(), "label": labels})
            stats = f.groupby("date").label.agg(["sum", "count"])
            means = (stats["sum"] + 30 * np.mean(labels)) / (stats["count"] + 30)
            return f.date.map(means).to_numpy()
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(wrong(self.y)[query], wrong(changed)[query])

    def test_injection_errors_rejected_before_encoding(self):
        p = RawFeaturePipeline(self.recipe)
        with self.assertRaises(ValueError):
            p.fit_transform(self.rows, self.y, self.groups, FitContext(155, 0),
                            encoder_splits=[self.splits[0]] * 5)
        recipe = dict(self.recipe, blocks=["date_count"])
        with self.assertRaises(ValueError):
            RawFeaturePipeline(recipe).fit_transform(self.rows, self.y, self.groups, FitContext(155, 0),
                                                    encoder_splits=self.splits)

    def test_query_values_cannot_refit_feature_state(self):
        fit, query = self.splits[0]
        p = RawFeaturePipeline(self.recipe)
        p.fit_transform(self.rows.iloc[fit], self.y[fit], self.groups[fit], FitContext(155, 0))
        expected, receipts = p.transform(self.rows.iloc[query]), copy.deepcopy(p.receipts_)
        poisoned = self.rows.iloc[query].copy()
        poisoned["loudness"] = 1e20
        poisoned[DATE] = "2099-12-31"
        p.transform(poisoned)
        pd.testing.assert_frame_equal(expected, p.transform(self.rows.iloc[query]), check_exact=True)
        self.assertEqual(receipts, p.receipts_)

    def test_worker_imports_do_not_load_model_runtimes(self):
        code = ("import sys,json; from songpop.phase2.raw_features import RawFeaturePipeline; "
                "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] "
                "in ['torch','lightgbm','xgboost','catboost'])))")
        result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
