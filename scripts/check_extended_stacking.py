"""Known-wrong and positive controls for the new outer-local meta fitting."""
import unittest
import numpy as np
import pandas as pd
from analysis.extended_stacking import meta_predictions
from analysis.extended_combiners import fit_pairwise, apply


class Boundary(unittest.TestCase):
    def test_pairwise_direction_and_shrinkage(self):
        y = np.array([0, 0, 1, 1])
        Z = np.array([[.2, .9], [.3, .1], [.7, .9], [.8, .1]])
        fitted = fit_pairwise(Z, y, 0.)
        regularized = fit_pairwise(Z, y, 10.)
        self.assertGreater(fitted['weights'][0], .99)
        self.assertLess(regularized['weights'][0], fitted['weights'][0])
        p = apply(fitted, Z)
        self.assertGreater(p[y == 1].min(), p[y == 0].max())
        self.assertAlmostEqual(sum(fitted['weights']), 1.)
        constant = fit_pairwise(np.full((4, 2), .5), y, .01)
        np.testing.assert_allclose(constant['weights'], [.5, .5])

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
