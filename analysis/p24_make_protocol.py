"""Writes configs/phase2/p2.4/protocol.json (plan section 18), from the project root, after P2.3 converged: the
frozen reference inventory (every complete fitted run of P2.2, P2.3 and P2.8 now in the log), the aliases resolved
by their rules from it, CatBoost and EBM from P2.3's handoff (analysis/p23_summary_output.txt), the P2.8 import
receipt, the comparisons, the queues and the re-tuning spaces. Fits nothing. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p24_make_protocol.py
--dev: no handoff (only for the isolated development copy without P2.3 runs)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from songpop import artifacts
from songpop.phase2 import feature_stage as fs, runs
from songpop.phase2.recipes import recipe_hash
DEV = "--dev" in sys.argv
ctx = runs.Context.load(Path("."))
sha = lambda p: artifacts.sha256_bytes(Path(p).read_bytes())
recs = runs.logged(ctx)
inv_recs = [r for r in recs if r["task_id"] in ("P2.2", "P2.3", "P2.8") and r["prediction_kind"] == "fitted"
            and r["selection_eligible"] and r["split_seeds"] == [155, 2026, 1029] and r["n_rows"] == 3863]
inventory = [{"run_id": r["run_id"], "bank_sha256": r["bank_sha256"]} for r in inv_recs]
pool = []
for r in inv_recs:
    bank, s, _ = runs.load_logged(ctx, r["run_id"])
    md = bank.metadata
    pool.append({"run_id": r["run_id"], "task_id": r["task_id"], "family": md["recipe"]["family"], "recipe": md["recipe"],
                 "recipe_id": md["recipe_id"], "recipe_hash": md["recipe_sha256"], "features": list(md["features"]),
                 "mean": float(bank.fold_auc.mean())})
by_name = {e["recipe"]["name"]: e for e in pool if e["task_id"] == "P2.8"}
m = json.load(open("configs/phase2/p2.8/manifest.json"))
routes = {"B12": ["ensemble_contribution"], "B14": ["smaller_useful", "competitive_refinement", "ensemble_contribution"]}
comps = []
for c in m["part_b"]["comparisons"]:
    ch, cp = by_name[c["challenger"]], by_name[c["comparator"]]
    rb = {r["run_id"]: r["bank_sha256"] for r in inv_recs}
    comps.append({"id": c["id"], "family": c["family"], "comparator_run_id": cp["run_id"], "comparator_bank_sha256": rb[cp["run_id"]],
                  "comparator_recipe": cp["recipe"], "challenger_run_id": ch["run_id"], "challenger_bank_sha256": rb[ch["run_id"]],
                  "challenger_recipe": ch["recipe"], "challenger_recipe_sha256": recipe_hash(ch["recipe"]), "routes": routes.get(c["id"], [])})
imports = {"task": "P2.8", "manifest": "configs/phase2/p2.8/manifest.json", "manifest_sha256": sha("configs/phase2/p2.8/manifest.json"),
           "judge": "analysis/p28_feature_checks.py", "judge_sha256": sha("analysis/p28_feature_checks.py"),
           "output": "analysis/p28_feature_checks_output.txt", "output_sha256": sha("analysis/p28_feature_checks_output.txt"),
           "h_run_id": by_name[m["part_b"]["allocation_state"]["H"]]["run_id"], "comparisons": comps}
if DEV:
    handoff = {"path": None, "sha256": None, "families": {"catboost": {"available": False, "reason": "dev copy"},
                                                          "ebm": {"available": False, "reason": "dev copy"}}}
else:
    text = Path("analysis/p23_summary_output.txt").read_text()
    h = json.loads(text[text.index("Handoff to P2.4"):].split("\n", 1)[1])
    handoff = {"path": "analysis/p23_summary_output.txt", "sha256": sha("analysis/p23_summary_output.txt"),
               "families": {f: ({"available": True, "run_id": v["best_standalone"]["run_id"]} if v["available"]
                                else {"available": False, "reason": v["reason"]}) for f, v in h["families"].items()}}
R = lambda rule, **kw: dict(rule=rule, **kw)
rules = {"L": R("named_run", tasks=["P2.2"], recipe_name="p22_lightgbm_b3_001"),
         "X": R("best_of_family", tasks=["P2.2"], family="xgboost"),
         "C": R("best_of_family", tasks=["P2.3"], family="catboost", fallback="X"),
         "M": R("best_of_family", tasks=["P2.3"], family="ebm"),
         "E": R("named_run", tasks=["P2.2"], recipe_name="p22_extra_trees_anchor"),
         "S": R("best_with_preprocessing", tasks=["P2.2"], family="logistic", base="F1", preprocessing="spline"),
         "K": R("best_of_family", tasks=["P2.2"], family="knn"),
         "HG": R("best_of_family", tasks=["P2.2"], family="hist_gb"),
         "R": R("best_with_column", tasks=["P2.2", "P2.3", "P2.8"], column="days")}
aliases = {}
for k, rule in rules.items():
    if k in ("C", "M"):
        fam = {"C": "catboost", "M": "ebm"}[k]
        hv = handoff["families"][fam]
        e = fs.alias_rule(rule, pool)
        if hv["available"]:
            assert e is not None and e["run_id"] == hv["run_id"], (k, e and e["run_id"], hv)
            aliases[k] = {"rule": rule, "run_id": hv["run_id"], "omitted_reason": None}
        elif k == "C":
            aliases[k] = {"rule": dict(rule, used_fallback=f"P2.3 handoff: catboost unavailable ({hv['reason']})"),
                          "run_id": fs.alias_rule(rules["X"], pool)["run_id"], "omitted_reason": None}
        else:
            aliases[k] = {"rule": rule, "run_id": None, "omitted_reason": f"P2.3 handoff: ebm unavailable ({hv['reason']})"}
    else:
        e = fs.alias_rule(rule, pool)
        aliases[k] = {"rule": rule, "run_id": e["run_id"], "omitted_reason": None}
C = lambda cid, alias, cmp=None, **ch: dict({"id": cid, "base": alias, "change": ch}, **({"comparator_change": cmp} if cmp else {}))
initial = [C("B15", "C", add=["date_count"]), C("B16", "C", add=["date_count", "date_te30"]), C("B17", "C", add=["music"]),
           C("B18", "C", add=["interactions"]), C("B19", "M", add=["music"]),
           C("B20", "L", add=["date_te10"]), C("B21", "L", add=["date_te30"]), C("B22", "L", add=["date_count", "date_te10"]),
           C("B23", "L", add=["date_te3"]), C("B24", "L", add=["date_count", "date_te3"]), C("B25", "L", add=["date_te100"]),
           C("B26", "L", add=["date_count", "date_te100"]), C("B27", "E", add=["date_count"]),
           C("B28", "E", add=["date_count", "date_te30"]),
           C("B29", "S", base="F2", preprocessing={"name": "spline_calendar"}),
           C("B30", "S", add=["cyclic"], preprocessing={"name": "spline_calendar"}),
           C("B31", "K", preprocessing={"name": "knn_blocks_v2", "audio_scaler": "robust"}),
           C("B32", "R", drop=["days"])]
F2CAL = {"base": "F2", "preprocessing": {"name": "spline_calendar"}}
ROB = {"preprocessing": {"name": "knn_blocks_v2", "audio_scaler": "robust"}}
smooth = [C("SK1", "S", F2CAL, base="F2", add=["date_count"], preprocessing={"name": "spline_calendar"}),
          C("SK2", "S", F2CAL, base="F2", add=["date_te30"], preprocessing={"name": "spline_calendar"}),
          C("SK3", "S", F2CAL, base="F2", add=["cyclic"], preprocessing={"name": "spline_calendar"}),
          C("SK4", "K", ROB, params={"metric": "manhattan"}, preprocessing={"name": "knn_blocks_v2", "audio_scaler": "robust", "normalization": "l1"})]
trees = [C("OT1", "E", add=["interactions"]), C("OT2", "X", add=["interactions"]), C("OT3", "E", add=["cyclic"]),
         C("OT4", "X", add=["cyclic"]), C("OT5", "E", add=["era_relative"]), C("OT6", "X", add=["era_relative"]),
         C("OT7", "HG", add=["interactions"]), C("OT8", "HG", add=["cyclic"])]
p22 = json.load(open("configs/phase2/p2.2/protocol.json")); p23 = json.load(open("configs/phase2/p2.3/protocol.json"))
fams = {}
for f in p22["families"] + p23["families"]:
    if f["family"] in ("lightgbm", "random_forest", "xgboost", "extra_trees", "hist_gb", "catboost", "ebm") and f["space"]:
        fams[f["family"]] = {k: f[k] for k in ("space", "fixed", "derived", "admissible", "types", "trees")}
fams["logistic"] = {"space": {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, "fixed": {}, "derived": [], "admissible": [],
                    "types": {"C": {"type": "log", "integer": False, "zero_step": None, "default": 1.0}}, "trees": None}
fams["knn"] = {"space": {"n_neighbors": [15, 30, 60, 120, 240, 480], "weights": ["uniform", "distance"]}, "fixed": {},
               "derived": [], "admissible": [],
               "types": {"n_neighbors": {"type": "log", "integer": True, "zero_step": None, "default": 5},
                         "weights": {"type": "categorical", "values": ["uniform", "distance"], "default": "uniform"}},
               "trees": None}
h_recipe = dict(by_name["p28_lightgbm_F2_anchor"]["recipe"], name="p24_h")
proto = {"schema": "feature_protocol_v1", "task_id": "P2.4", "name": "p24", "split_seeds": [155, 2026, 1029],
         "cutoff_utc": "2026-10-31T11:00:00+00:00", "retune_budget_end_utc": "2026-10-31T12:00:00+00:00",
         "reference_p21_run_id": p23["reference_p21_run_id"], "h_recipe": h_recipe, "reference_inventory": inventory,
         "p23_handoff": handoff, "imports": imports, "aliases": aliases, "initial": initial,
         "expansion": {"size": 16, "max_comparisons": 96,
                       "queue": [["dates", 6], ["combinations", 2], ["smooth_knn", 2], ["removals", 4], ["other_trees", 1], ["weights", 1]],
                       "date_families": ["xgboost", "catboost", "hist_gb", "ebm"], "date_bases_per_batch": 2,
                       "smooth_knn": smooth, "other_trees": trees},
         "retune": {"max_units": 4, "max_per_family": 2, "by_standalone": 2, "batch": 16, "local": 12, "broad": 4,
                    "max_batches": 2, "seed_base": 2026103100, "families": fams}}
fs.load_protocol(proto)
Path("configs/phase2/p2.4").mkdir(parents=True, exist_ok=True)
open("configs/phase2/p2.4/protocol.json", "w").write(json.dumps(proto, indent=1) + "\n")
print("protocol written:", len(inventory), "reference runs;", {k: v["run_id"] or v["omitted_reason"] for k, v in aliases.items()})
