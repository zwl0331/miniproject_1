"""Writes configs/phase2/p2.4/timing.json (plan section 18.9) from the protocol's resolved aliases: score-free timing
cases for the feature recipes. Fits nothing. From the project root, after the protocol is written:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p24_make_timing.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts import run_experiments as rx
from songpop.phase2 import feature_stage as fs, runs
ctx = runs.Context.load(Path("."))
proto = rx.load_feature_protocol(ctx, "P2.4")[0]
pool = rx.feature_pool(ctx, proto, rx.FeatureLog(ctx, "P2.4"))
a = fs.resolve_aliases(proto, pool)
def ch(alias, change, name):
    return fs.changed(a[alias]["recipe"], change, name)
cases = [("representative", ch("L", {"add": ["date_count", "date_te30"]}, "p24_time_lightgbm_dates")),
         ("representative", ch("C", {"add": ["date_count", "date_te30"]}, "p24_time_catboost_dates")),
         ("representative", ch("S", {"base": "F2", "add": ["date_count"], "preprocessing": {"name": "spline_calendar"}}, "p24_time_logistic_calendar")),
         ("representative", ch("K", {"preprocessing": {"name": "knn_blocks_v2", "audio_scaler": "robust"}}, "p24_time_knn_robust"))]
if a["C"]["family"] == "catboost":
    cases.append(("expensive", ch("C", {"add": ["date_count", "date_te30"], "params": {"depth": 7, "iterations": 2000, "learning_rate": 0.05, "boosting_type": "Ordered"}}, "p24_time_catboost_corner")))
if a["M"] is not None:
    cases.append(("representative", ch("M", {"add": ["music"]}, "p24_time_ebm_music")))
spec = {"schema": "timing_v1", "cases": [{"case": c, "recipe": r} for c, r in cases]}
Path("configs/phase2/p2.4").mkdir(parents=True, exist_ok=True)
open("configs/phase2/p2.4/timing.json", "w").write(json.dumps(spec, indent=1) + "\n")
print([(c, r["name"], r["family"]) for c, r in cases])
