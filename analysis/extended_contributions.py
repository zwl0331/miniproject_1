"""Fixed probability/logit additions and matched-channel replacements."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from analysis.extended_summary import ROOT, OUT, BASE
from analysis.fast_exploration import load_arrays, score, write_json
from songpop.data import sha256_file


def main():
    base = load_arrays(BASE)
    channels = {family: load_arrays(ROOT / f"artifacts/fast/{name}/predictions.npz") for family, name in
        (("neural", "mlp_w128_a1.0_e300_standard_seeds012"),
         ("lightgbm", "f2_avg_lgb_zscore5"), ("extra_trees", "f2_avg_et5"))}
    destination = OUT / "contributions.json"
    cache = json.loads(destination.read_text()) if destination.exists() else []
    done = {row["id"]: row["prediction_sha256"] for row in cache}
    results = list(cache)
    for parent in (OUT, ROOT / "artifacts/neural", ROOT / "artifacts/fast"):
        for path in sorted(parent.glob("x3_*/result.json")):
            result = json.loads(path.read_text())
            config = result["config"]
            family = config.get("family", config.get("recipe", {}).get("family"))
            prediction_path = path.with_name("predictions.npz")
            digest = sha256_file(prediction_path)
            if result["id"] in done:
                if digest != done[result["id"]]:
                    raise ValueError("a scored prediction bank changed")
                continue
            a = load_arrays(prediction_path)
            for key in ("train_row_id", "y", "group_id", "fold_id", "split_seeds", "test_id"):
                if not np.array_equal(a[key], base[key]):
                    raise ValueError(f"{path} differs in {key}")
            variants = [(f"add_{w:g}", w, (1-w)*base["oof"] + w*a["oof"])
                        for w in (.05, .1, .25, .5)]
            variants += [(f"logit_{w:g}", w, expit((1-w)*logit(np.clip(base["oof"], 1e-6, 1-1e-6)) +
                         w*logit(np.clip(a["oof"], 1e-6, 1-1e-6)))) for w in (.1, .25)]
            channel = "neural" if family in ("tabm", "mlp", "resnet", "ft_transformer", "retrieval") else family
            if channel in channels:
                w = .1 if channel == "neural" else .25
                variants.append(("replace_" + channel, w,
                                 base["oof"] - w*channels[channel]["oof"] + w*a["oof"]))
            for comparison, w, predictions in variants:
                auc = score(a["y"], a["fold_id"], predictions)
                delta = auc.mean(axis=1) - base["fold_auc"].mean(axis=1)
                results.append(dict(id=result["id"], family=family, comparison=comparison,
                    weight=w, cv=float(auc.mean()), delta=float(delta.mean()),
                    per_seed=auc.mean(axis=1).tolist(), delta_per_seed=delta.tolist(),
                    positive_splits=int((delta > 0).sum()), prediction_sha256=digest,
                    source=str(prediction_path.relative_to(ROOT))))
    results.sort(key=lambda row: row["cv"], reverse=True)
    write_json(destination, results)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "contributions.csv", index=False)
    print(frame.drop_duplicates("id").head(12)[["id", "comparison", "cv", "delta", "positive_splits"]].to_string(index=False))


if __name__ == "__main__":
    main()
