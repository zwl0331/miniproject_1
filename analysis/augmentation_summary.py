"""Compare the four bounded augmentation settings with the preserved best bank."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from analysis.fast_exploration import ROOT, load_arrays, score, write_json

OUT = ROOT / "artifacts/augmentation"
REFERENCE = "artifacts/fast/x3_fixed_hybrid_retrieval5/predictions.npz"
HYBRID = "artifacts/fast/x3_fixed_hybrid_pairlogit_d4_lr0p03_l210p0/predictions.npz"


def variants(base, candidate, hybrid, family):
    result = {f"add_{w:g}": (1-w)*base + w*candidate for w in (.05, .1)}
    result["logit_0.1"] = expit(.9*logit(np.clip(base,1e-6,1-1e-6)) + .1*logit(np.clip(candidate,1e-6,1-1e-6)))
    if family == "retrieval":
        result["replace_retrieval"] = .95*hybrid + .05*candidate
    return result


def compare(path, original_path):
    base, hybrid = load_arrays(ROOT/REFERENCE), load_arrays(ROOT/HYBRID)
    bank, original = load_arrays(path), load_arrays(original_path)
    result = json.loads(path.with_name("result.json").read_text())
    family = result["config"].get("family", result["config"].get("recipe",{}).get("family"))
    for a in (bank, original):
        for key in ("train_row_id","y","group_id","fold_id","split_seeds","test_id"):
            if not np.array_equal(a[key],base[key]):
                raise ValueError(f"augmentation comparison mismatch in {key}")
    controls = {name:score(base['y'],base['fold_id'],p) for name,p in
                variants(base['oof'],original['oof'],hybrid['oof'],family).items()}
    rows=[]
    for name,p in variants(base['oof'],bank['oof'],hybrid['oof'],family).items():
        auc=score(base['y'],base['fold_id'],p);delta=auc.mean(1)-base['fold_auc'].mean(1)
        rows.append(dict(id=result['id'],family=family,comparison=name,cv=float(auc.mean()),
            delta=float(delta.mean()),per_seed=auc.mean(1).tolist(),delta_per_seed=delta.tolist(),
            positive_splits=int((delta>0).sum()),standalone=result['cv'],
            delta_versus_unaugmented_control=float((auc-controls[name]).mean()),
            path=str(path.relative_to(ROOT))))
    return rows


def main():
    queue=json.loads((OUT/'queue.json').read_text()); rows=[]; remaining=[]
    for config in queue:
        path=ROOT/'artifacts/neural'/config['id']/'predictions.npz'
        if not path.with_name('result.json').exists():
            remaining.append(config['id']);continue
        family=config['recipe']['family']
        original='x3_tabm_auc' if family=='tabm' else 'x3_retrieval_c16_w64_scalar'
        rows += compare(path,ROOT/'artifacts/neural'/original/'predictions.npz')
    rows.sort(key=lambda x:x['cv'],reverse=True)
    write_json(OUT/'summary.json',dict(reference=REFERENCE,baseline_cv=.8446492316936202,
        completed=len(queue)-len(remaining),remaining=remaining,results=rows,
        limitation='Exploratory reused grouped folds; initial augmentation configs use estimator seed0.'))
    if rows:
        with (OUT/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('Complete',len(queue)-len(remaining),'/4; remaining',remaining)
    for row in rows[:8]:
        print(row['id'],row['comparison'],f"CV={row['cv']:.9f}",f"delta={row['delta']:+.9f}",
              f"split signs={row['positive_splits']}/3",f"vs unaugmented control={row['delta_versus_unaugmented_control']:+.9f}")


if __name__=='__main__':
    main()
