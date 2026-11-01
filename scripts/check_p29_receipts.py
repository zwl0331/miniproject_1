"""P2.9 receipt gates: historical compatibility and corrupted-state rejection."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from songpop.evaluate import PredictionBank
from songpop.phase2 import banks, procedures
from songpop.phase2.encoders import FitContext
from songpop.phase2.receipts import legacy_tabm

ROOT = Path(__file__).resolve().parents[1]
TABM = ROOT / "artifacts/phase2/bank/p2_p25_tabm_anchor_seeds155-2026-1029_984d3248a783.npz"


class Receipts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = banks.load_bank_v2(TABM)
        cls.recipe = cls.bank.metadata["recipe"]
        cls.receipt = cls.bank.metadata["fit_receipts"][0]
        fitting = cls.bank.fold_id[0] != 0
        cls.args = (cls.bank.train_row_id[fitting], cls.bank.y[fitting], cls.bank.group_id[fitting], FitContext(155, 0))

    def test_historical_receipts_remain_valid(self):
        legacy_tabm(self.recipe, self.receipt, *self.args)
        banks.validate_v2(self.bank)

    def test_corrupt_neural_receipts_are_refused(self):
        st = self.receipt["tabm"][0]
        changes = {
            "chosen_epoch": st["chosen_epoch"] + 1,
            "best_epoch": st["best_epoch"] + 1,
            "epochs_evaluated": st["epochs_evaluated"] + 1,
            "stopping_updates": st["stopping_updates"] + 1,
            "final_updates": st["final_updates"] + 1,
            "estimator_seed": 1,
            "inner_train_rows": st["inner_train_rows"] + 1,
            "inner_train_rows_sha256": "0" * 64,
            "inner_validation_rows_sha256": "0" * 64,
            "stopping_preprocessing_rows_sha256": self.receipt["fitting_rows_sha256"],
            "final_preprocessing_rows_sha256": "0" * 64,
            "stopping_preprocessing_digest": "bad",
            "final_preprocessing_digest": None,
            "encoded_columns": ["duplicate", "duplicate"],
            "versions": {"torch": "", "tabm": "0.0.3"},
            "policy": "unknown",
            "seeds": dict(st["seeds"], stop_init=st["seeds"]["stop_init"] + 1),
            "validation_losses": [float("nan")] * st["epochs_evaluated"],
            "stopped_by": "max_epochs" if st["stopped_by"] == "patience" else "patience",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt["tabm"][0][field] = value
                with self.assertRaises(ValueError):
                    legacy_tabm(self.recipe, receipt, *self.args)
        for field in ("chosen_epoch", "stopping_updates", "epochs_evaluated", "estimator_seed"):
            with self.subTest(boolean_field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt["tabm"][0][field] = True
                with self.assertRaises(ValueError):
                    legacy_tabm(self.recipe, receipt, *self.args)

    def test_full_bank_validation_reaches_receipts(self):
        bank = copy.deepcopy(self.bank)
        bank.metadata["fit_receipts"][0]["tabm"][0]["chosen_epoch"] += 1
        with self.assertRaisesRegex(ValueError, "chosen epoch"):
            banks.validate_v2(bank)

    def test_missing_seed_and_reused_fold_receipts_are_refused(self):
        for value in ([], [self.receipt["tabm"][0]] * 2):
            with self.assertRaises(ValueError):
                legacy_tabm(self.recipe, dict(self.receipt, tabm=value), *self.args)
        with self.assertRaises(ValueError):
            legacy_tabm(self.recipe, self.bank.metadata["fit_receipts"][1], *self.args)

    def test_older_procedure_banks_and_regularization_rejections(self):
        seen = set()
        for path in sorted((ROOT / "artifacts/phase2/bank").glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                md = json.loads(str(z["metadata_json"]))
                if md.get("schema") != procedures.PROCEDURE_SCHEMA:
                    continue
                arrays = {k: z[k] for k in PredictionBank.ARRAYS}
            bank = PredictionBank(**arrays, metadata=md)
            procedures.validate_procedure_metadata(bank)
            kind = md["procedure"]["setting"][0]
            if kind in seen:
                continue
            seen.add(kind)
            key = "lambda" if kind == "constrained" else "C"
            for value in (None, True, float("nan"), md["procedure"]["setting"][1] + 0.01):
                with self.subTest(kind=kind, value=value):
                    altered = copy.deepcopy(bank)
                    altered.metadata["states"][0][key] = value
                    with self.assertRaisesRegex(ValueError, "declared setting"):
                        procedures.validate_procedure_metadata(altered)
        self.assertEqual(seen, {"constrained", "stacker"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
