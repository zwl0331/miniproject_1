"""P2.5 runtime pilot behind Finding 024, score-free: (1) the OpenMP conflict, by import order, each in a fresh
process (exit code 139 is a segmentation fault); (2) the TabM worker's repeatability: the representative
configuration fitted on one production training part in two separate fitting processes, prediction bytes compared,
and the chosen epochs and elapsed times printed. Nothing is scored. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p25_runtime_pilot.py > analysis/p25_runtime_pilot_output.txt
"""
import hashlib
import json
import os
import subprocess
import sys
import time

PY = sys.executable
PROBE = "import torch\ntorch.set_num_threads(4)\na = torch.randn(500, 500)\nprint(float((a @ a).sum()) != 0.0)"
ORDERS = ["import lightgbm", "import xgboost", "import catboost", "import interpret.glassbox", "import sklearn.ensemble",
          None]
FIT = """
import hashlib, json, sys, time
sys.path.insert(0, ".")
import numpy as np
from songpop.phase2 import runs
from songpop.phase2.encoders import FitContext
from songpop.phase2.engine import fit_recipe
rec = json.loads(sys.argv[1])
ctx = runs.Context.load(".")
fold = ctx.fold_id([155])[0]
tr, va = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
t = time.time()
f = fit_recipe(rec, ctx.rows.iloc[tr], ctx.data.y[tr], ctx.groups[tr], FitContext(155, 0))
p = f.predict_high(ctx.rows.iloc[va])
q = f.predict_high(ctx.test_rows)
r = f.receipt["tabm"][0] if "tabm" in f.receipt else {}
print(json.dumps({"seconds": round(time.time() - t, 1), "chosen_epoch": r.get("chosen_epoch"),
                  "epochs_evaluated": r.get("epochs_evaluated"), "stopped_by": r.get("stopped_by"),
                  "lightgbm_loaded": "lightgbm" in sys.modules, "torch_loaded": "torch" in sys.modules,
                  "validation_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
                  "test_sha256": hashlib.sha256(q.tobytes()).hexdigest()}))
"""


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    print("1. torch's first matrix product after another library, each in a fresh process (exit code; 139 is SIGSEGV):")
    for first in ORDERS:
        code = subprocess.run([PY, "-c", (first + "\n" if first else "") + PROBE], env=env, capture_output=True).returncode
        print(f"   {first or '(torch alone)':28s} then torch: exit {code if code >= 0 else 128 - code}")
    code = subprocess.run([PY, "-c", "import torch\nimport lightgbm\n" + PROBE], env=env, capture_output=True).returncode
    print(f"   {'import torch first, then lightgbm':28s}: exit {code if code >= 0 else 128 - code}")
    rec = {"schema": "recipe_v1", "name": "p25_pilot_representative", "family": "tabm",
           "params": {"n_blocks": 2, "d_block": 128, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001,
                      "batch_size": 256}, "base": "F2", "blocks": [], "drop": [],
           "preprocessing": {"name": "tabm_numeric", "scaler": "standard"}, "training_weights": "none",
           "estimator_seeds": [0], "iterations": "early_stop_refit_v1", "deployment": "fold_average"}
    print("2. the representative TabM configuration on the training part of split seed 155, fold 0, fitted through the "
          "worker in two separate fitting processes (LightGBM loaded in each):")
    outs = []
    for i in range(2):
        t = time.time()
        res = subprocess.run([PY, "-c", FIT, json.dumps(rec)], env=env, capture_output=True, text=True, check=True)
        outs.append(json.loads(res.stdout.strip().splitlines()[-1]))
        print(f"   process {i + 1}: {json.dumps(outs[-1])} (wall {time.time() - t:.1f} s)")
    same = all(outs[0][k] == outs[1][k] for k in ("validation_sha256", "test_sha256", "chosen_epoch"))
    print(f"   prediction bytes and chosen epoch identical across processes: {same}")


if __name__ == "__main__":
    main()
