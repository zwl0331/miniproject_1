"""Known-wrong and positive controls for the new outer-local meta fitting."""
import unittest
import numpy as np
import pandas as pd
from analysis.extended_stacking import meta_predictions


class Boundary(unittest.TestCase):
    def test_labels_stay_inside_meta_training(self):
        groups = np.repeat(np.arange(30), 2)
        rows = pd.DataFrame({"x": np.arange(60.)}, index=np.arange(1000, 1060))
        y = np.tile([0, 1], 30)
        folds = groups % 5
        calls = []
        def honest(channel, train, labels, train_groups, context, query):
            self.assertFalse(set(train.index) & set(query.index))
            self.assertTrue(set(train.index) <= set(rows.index))
            calls.append(context.fold)
            return np.full(len(query), np.mean(labels))
        before = meta_predictions({}, rows, y, groups, folds, "test", honest)
        poisoned = y.copy(); poisoned[folds == 0] = 1
        after = meta_predictions({}, rows, poisoned, groups, folds, "test", honest)
        np.testing.assert_array_equal(before[folds == 0], after[folds == 0])
        self.assertTrue(np.any(before[folds != 0] != after[folds != 0]))
        self.assertEqual(calls, list(range(5)) * 2)
        # A deliberately wrong global-label fit fails the same held-out check.
        def leaky(labels):
            return lambda channel, train, yy, gg, ctx, query: np.full(len(query), np.mean(labels))
        wrong_before = meta_predictions({}, rows, y, groups, folds, "test", leaky(y))
        wrong_after = meta_predictions({}, rows, poisoned, groups, folds, "test", leaky(poisoned))
        self.assertFalse(np.array_equal(wrong_before[folds == 0], wrong_after[folds == 0]))

    def test_split_track_rejected(self):
        groups = np.repeat(np.arange(30), 2)
        rows = pd.DataFrame({"x": np.arange(60.)})
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            meta_predictions({}, rows, np.tile([0, 1], 30), groups, np.arange(60) % 5, "bad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
