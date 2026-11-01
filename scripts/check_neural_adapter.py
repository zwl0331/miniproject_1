"""Small end-to-end checks for the fast neural adapter, including CUDA."""
import copy
import os
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

from songpop.data import load_dataset
from songpop.phase2.engine import raw_rows
from songpop.phase2.encoders import FitContext, inner_group_splits
from songpop.phase2 import raw_neural as N, raw_recipes as R, tabm as T
from songpop.phase2.neural_architectures import specification

ROOT = Path(__file__).resolve().parents[1]
CASES = [(family, rep) for family in ("tabm", "mlp", "resnet")
         for rep in ("scalar", "ple", "periodic")] + [("ft_transformer", "scalar")]


class AdvancedNeural(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_dataset(ROOT)
        cls.rows = raw_rows(data.train_eligible, data.row_ids).iloc[:200].copy()
        cls.rows.index = np.arange(10000, 10200)
        cls.rows["track_album_release_date"] = [f"{2000 + i % 20}-01-01" for i in range(200)]
        cls.y = np.tile([0, 1], 100)
        cls.groups = np.repeat(np.arange(100), 2)
        cls.context = FitContext(155, 0)

    @classmethod
    def tearDownClass(cls):
        N.Worker.close_all()

    def recipe(self, family, representation, device):
        return R.make_recipe(family, representation=representation, device=device,
            blocks=["date_count", "date_te3"], params={"max_epochs": 3, "patience": 2,
                "d_block": 32, "n_blocks": 1, "batch_size": 64})

    def test_all_architectures_cpu_and_cuda_fit_predict(self):
        devices = ("cpu", "cuda") if os.environ.get("NEURAL_TEST_CUDA") == "1" else ("cpu",)
        for device in devices:
            for family, rep in CASES:
                with self.subTest(device=device, family=family, representation=rep):
                    recipe = self.recipe(family, rep, device)
                    fitted = N.fit_raw(recipe, self.rows, self.y, self.groups, self.context)
                    query = self.rows.iloc[:8]
                    pred = fitted.predict_proba(query)
                    self.assertEqual(pred.shape, (8, 2))
                    self.assertEqual(fitted.classes_.tolist(), [0, 1])
                    self.assertTrue(np.isfinite(pred).all())
                    self.assertTrue(((pred >= 0) & (pred <= 1)).all())
                    np.testing.assert_allclose(pred.sum(axis=1), 1, atol=0, rtol=0)
                    self.assertGreater(np.ptp(pred[:, 1]), 0)
                    self.assertEqual(fitted.receipt["raw_neural"]["subfits"]["encoder_fits"], 12)
                    restored = pickle.loads(pickle.dumps(fitted))
                    np.testing.assert_array_equal(restored.predict_proba(query), pred)
                    # A query can neither mutate fitted transforms nor embedding bins.
                    before = copy.deepcopy(fitted.model_spec)
                    poisoned = query.copy()
                    poisoned["loudness"] += 10000
                    fitted.predict_high(poisoned)
                    self.assertEqual(before, fitted.model_spec)
                    np.testing.assert_array_equal(fitted.predict_proba(query), pred)
                    print(f"PASS {device} {family} {rep}", flush=True)
        self.assertNotIn("torch", sys.modules)

    def test_ple_stopping_labels_and_features_are_isolated(self):
        recipe = self.recipe("tabm", "ple", "cpu")
        fit, valid = T.inner_split(self.y, self.groups, 5, T.stage_seeds(self.context, 0)["inner_split"])
        diagnostics = {"stop_split": (fit, valid), "stop_encoder_splits":
            inner_group_splits(self.y[fit], self.groups[fit], self.context.inner_seed("date_te"))}
        request = {"op": "trajectory", "recipe": recipe, "rows": self.rows, "y": self.y,
            "groups": self.groups, "context": self.context, "diagnostics": diagnostics, "epochs": 3, "seed": 0}
        worker = N.Worker.for_device("cpu")
        original = worker.call(request)
        y = self.y.copy()
        y[valid] = 1 - y[valid]
        flipped = worker.call(dict(request, y=y))
        for key in ("trajectory", "feature_state_sha256", "preprocessing_sha256", "model_spec_sha256"):
            self.assertEqual(original[key], flipped[key])
        self.assertNotEqual(original["losses"], flipped["losses"])
        rows = self.rows.copy()
        rows.iloc[valid, rows.columns.get_loc("loudness")] += 1000
        poisoned = worker.call(dict(request, rows=rows))
        self.assertEqual(original["model_spec_sha256"], poisoned["model_spec_sha256"])
        self.assertEqual(original["trajectory"], poisoned["trajectory"])
        # Positive control: allowed training data changes must change the bins.
        rows = self.rows.copy()
        rows.iloc[fit[0], rows.columns.get_loc("loudness")] += 1000
        allowed = worker.call(dict(request, rows=rows))
        self.assertNotEqual(original["model_spec_sha256"], allowed["model_spec_sha256"])

    def test_constant_bins_and_invalid_recipes(self):
        recipe = self.recipe("tabm", "ple", "cpu")
        spec = specification(recipe, np.zeros((10, 2), dtype="float32"))
        self.assertEqual(spec["bins"], [[-0.5, 0.5], [-0.5, 0.5]])
        for kwargs in ({"family": "unknown"}, {"family": "ft_transformer", "representation": "ple"},
                       {"family": "ft_transformer", "params": {"d_block": 33}},
                       {"family": "mlp", "params": {"k": 16}},
                       {"representation": "ple", "embedding": {"n_bins": 1}}):
            with self.assertRaises(ValueError):
                R.make_recipe(**kwargs)

    def test_resnet_keeps_singleton_final_batch(self):
        # 200 % 199 == 1. BatchNorm uses fitted running statistics for that row.
        recipe = self.recipe("resnet", "ple", "cpu")
        recipe["params"]["batch_size"] = 199
        fitted = N.fit_raw(recipe, self.rows, self.y, self.groups, self.context)
        receipt = fitted.receipt["tabm"][0]
        self.assertEqual(receipt["final_updates"], 2 * receipt["chosen_epoch"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
