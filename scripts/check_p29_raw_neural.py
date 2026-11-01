"""Bounded scalar raw-neural checks before full-data equivalence gates."""
import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from songpop.phase2 import neural_backend as B, raw_neural as N, raw_recipes as R, tabm as T
from songpop.phase2.encoders import FitContext, inner_group_splits
from songpop.phase2.features import FeaturePipeline
from songpop.phase2.banks import load_bank_v2
from songpop.phase2.runs import Context

ROOT = Path(__file__).resolve().parents[1]

LEAKY_CHILD = """
import json, pickle, sys
import numpy as np
from songpop.phase2 import neural_backend as B, tabm as T
from songpop.phase2.raw_features import RawFeaturePipeline
recipe, rows, y, groups, context, diagnostic, flipped = pickle.load(sys.stdin.buffer)
correct_prepare = B.prepare
def leaky_prepare(recipe, rows, y, groups, context, diagnostic=None, **options):
    result = correct_prepare(recipe, rows, y, groups, context, diagnostic, **options)
    features = RawFeaturePipeline(recipe)
    X = features.fit_transform(rows, y, groups, context, encoder_splits=diagnostic['final_encoder_splits'])
    prep = T.TabMPreprocessor(recipe['preprocessing']['scaler']).fit(X)
    fit, valid = result['fit'], result['valid']
    result.update(stop_features=features, stop_prep=prep, Xs=X.iloc[fit], Xv=X.iloc[valid],
                  Zs=prep.transform(X.iloc[fit]), Zv=prep.transform(X.iloc[valid]))
    return result
B.prepare = leaky_prepare
torch = B.configure('cpu')
assert not {'lightgbm', 'xgboost', 'songpop.artifacts'} & set(sys.modules)
a = B.trajectory(torch, recipe, rows, y, groups, context, diagnostic, 3, 0)
b = B.trajectory(torch, recipe, rows, flipped, groups, context, diagnostic, 3, 0)
print(json.dumps({'a': a, 'b': b}))
"""


class RawNeural(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx = Context.load(ROOT)
        cls.rows = ctx.rows.iloc[:400].copy()
        cls.rows.index = np.arange(10000, 10400)
        cls.rows["track_album_release_date"] = [f"{2000 + i % 20}-01-01" for i in range(400)]
        cls.y = np.tile([0, 1], 200)
        cls.groups = np.repeat(np.arange(200), 2)
        cls.context = FitContext(155, 0)
        v1 = load_bank_v2(ROOT / "artifacts/phase2/bank/p2_p25_tabm_anchor_seeds155-2026-1029_984d3248a783.npz").metadata["recipe"]
        original = copy.deepcopy(v1)
        cls.recipe = R.from_legacy(v1)
        assert original == v1
        cls.recipe["params"].update(n_blocks=2, d_block=32, dropout=0.1, learning_rate=0.003,
                                   weight_decay=1e-4, batch_size=64, max_epochs=4, patience=2)
        cls.fitted = N.fit_raw(cls.recipe, cls.rows, cls.y, cls.groups, cls.context)

    @classmethod
    def tearDownClass(cls):
        N.Worker.close_all()
        T.Worker.close()

    def test_scalar_cpu_matches_legacy_training_receipt_and_predictions(self):
        X = FeaturePipeline(self.recipe).fit_transform(self.rows, self.y, self.groups, self.context)
        old = T.TabMClassifier(**self.recipe["params"], scaler="standard", random_state=0).fit(
            X, self.y, groups=self.groups, context=self.context)
        np.testing.assert_array_equal(old.predict_proba(X)[:, 1], self.fitted.predict_high(self.rows))
        np.testing.assert_array_equal(old.transform_inputs(X), self.fitted.model_inputs(self.rows))
        self.assertEqual(old.receipt_, self.fitted.receipt["tabm"][0])
        self.assertNotIn("torch", sys.modules)

    def test_seed_average_equals_separate_fits_exactly(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["estimator_seeds"] = [0, 1]
        together = N.fit_raw(recipe, self.rows, self.y, self.groups, self.context)
        recipe["estimator_seeds"] = [1]
        one = N.fit_raw(recipe, self.rows, self.y, self.groups, self.context)
        expected = np.mean([self.fitted.predict_high(self.rows), one.predict_high(self.rows)], axis=0)
        np.testing.assert_array_equal(together.predict_high(self.rows), expected)
        self.assertEqual(together.receipt["raw_neural"]["subfits"]["feature_fits"], 2)
        self.assertEqual(together.receipt["raw_neural"]["subfits"]["network_trainings"], 4)

    def test_fixed_assignment_stopping_isolation_and_positive_control(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["blocks"] = ["date_count", "date_te3"]
        fit, valid = T.inner_split(self.y, self.groups, 5, T.stage_seeds(self.context, 0)["inner_split"])
        encoder = inner_group_splits(self.y[fit], self.groups[fit], self.context.inner_seed("date_te"))
        diagnostics = {"stop_split": (fit, valid), "stop_encoder_splits": encoder,
                       "final_encoder_splits": inner_group_splits(self.y, self.groups, self.context.inner_seed("date_te"))}
        request = {"op": "trajectory", "recipe": recipe, "rows": self.rows, "y": self.y, "groups": self.groups,
                   "context": self.context, "diagnostics": diagnostics, "epochs": 3, "seed": 0}
        worker = N.Worker.for_device("cpu")
        original = worker.call(request)
        flipped = self.y.copy()
        flipped[valid] = 1 - flipped[valid]
        altered = worker.call(dict(request, y=flipped))
        invariant = ["trajectory", "updates", "split_sha256", "feature_state_sha256", "preprocessing_sha256",
                     "Xs_sha256", "Xv_sha256", "Zs_sha256", "Zv_sha256"]
        for key in invariant:
            self.assertEqual(original[key], altered[key], key)
        self.assertNotEqual(original["losses"], altered["losses"])
        permitted = self.y.copy()
        permitted[fit[0]] = 1 - permitted[fit[0]]
        positive = worker.call(dict(request, y=permitted))
        self.assertNotEqual(original["Xv_sha256"], positive["Xv_sha256"])
        # The known-wrong worker diagnostic fits D-wide transforms before
        # stopping. Fixed assignments and the real training loop must expose it.
        bad = subprocess.run([sys.executable, "-c", LEAKY_CHILD], cwd=ROOT,
            input=pickle.dumps((recipe, self.rows, self.y, self.groups, self.context, diagnostics, flipped)),
            capture_output=True, check=True, timeout=120)
        wrong = json.loads(bad.stdout)
        with self.assertRaises(AssertionError):
            self.assertEqual(wrong["a"]["trajectory"], wrong["b"]["trajectory"])
        # Full D-arm fit exercises the same worker path and its new final state.
        fitted = N.fit_raw(recipe, self.rows, self.y, self.groups, self.context, diagnostics=diagnostics)
        self.assertTrue(np.isfinite(fitted.predict_high(self.rows.iloc[:20])).all())
        self.assertTrue(fitted.receipt["raw_neural"]["diagnostic_assignments"])
        self.assertEqual(fitted.receipt["raw_neural"]["subfits"]["encoder_fits"], 12)

    def test_duplicate_queries_and_worker_restart_are_exact(self):
        query = self.rows.iloc[np.r_[np.arange(300), 0]]
        p = self.fitted.predict_high(query)
        self.assertEqual(p[0], p[-1])
        self.assertEqual(p[0], self.fitted.predict_high(self.rows.iloc[[0]])[0])
        N.Worker.close_all()
        np.testing.assert_array_equal(self.fitted.predict_high(query), p)

    def test_query_poisoning_cannot_refit_state(self):
        before = (B.object_digest(B.feature_state(self.fitted.features)), self.fitted.prep.digest_)
        query = self.rows.copy()  # keep the category vocabulary fixed in the deliberately wrong control
        query["loudness"] += 1000
        self.fitted.predict_high(query)
        after = (B.object_digest(B.feature_state(self.fitted.features)), self.fitted.prep.digest_)
        self.assertEqual(before, after)
        wrong = copy.deepcopy(self.fitted)
        transform = wrong.prep.transform
        def leaky(X):
            wrong.prep.fit(X)
            return transform(X)
        with mock.patch.object(wrong.prep, "transform", leaky):
            wrong.predict_high(query)
        with self.assertRaises(AssertionError):
            self.assertEqual(wrong.prep.digest_, before[1])

    def test_both_sides_reject_invalid_diagnostic_splits(self):
        diagnostics = {"stop_split": (np.arange(300), np.arange(200, 400))}
        with self.assertRaises(ValueError):
            N.fit_raw(self.recipe, self.rows, self.y, self.groups, self.context, diagnostics=diagnostics)
        with self.assertRaisesRegex(RuntimeError, "partition"):
            N.Worker.for_device("cpu").call({"op": "fit", "recipe": self.recipe, "rows": self.rows,
                "y": self.y, "groups": self.groups, "context": self.context, "diagnostics": diagnostics})

    def test_schema_rejects_unimplemented_fields_and_legacy_conversion_is_pure(self):
        for alter in ({"device": "auto"}, {"representation": "ple"}, {"extra": 1}):
            recipe = copy.deepcopy(self.recipe)
            recipe["adapter"].update(alter)
            with self.assertRaises(ValueError):
                R.validate(recipe)
        for field in ("threads", "inner_folds", "max_epochs"):
            recipe = copy.deepcopy(self.recipe)
            recipe["params"][field] = True
            with self.assertRaises(ValueError):
                R.validate(recipe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
