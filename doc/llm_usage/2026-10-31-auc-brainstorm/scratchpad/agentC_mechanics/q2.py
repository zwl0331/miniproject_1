import pipe, traceback, time, json, numpy as np
F2 = pipe.F2_COLS
Lp_blocks = [pipe.block_date_count, pipe.make_block_date_te(3.0)]
days_idx = F2.index('days')
cat_mbf = [255]*len(F2) + [4095]        # only the categorical date column gets a large bin budget
def mbf_days(n_extra=0, days=4095):
    v = [255]*len(F2) + [255]*n_extra; v[days_idx] = days; return v

# pick the best A setting on L from results.jsonl (fallback: default)
best = None
try:
    recs = [json.loads(l) for l in open(pipe.RESULTS)]
    A = [r for r in recs if r['name'].startswith('A_L_')]
    if A:
        best = max(A, key=lambda r: r['mean'])
        print('best A on L:', best['name'], round(best['mean'],4), flush=True)
except Exception:
    traceback.print_exc()
def best_over(n_extra):
    if best is None: return {}
    p = best['params']; o = {}
    if 'max_bin' in p: o['max_bin'] = p['max_bin']
    if 'min_data_in_bin' in p: o['min_data_in_bin'] = p['min_data_in_bin']
    if 'max_bin_by_feature' in p: o['max_bin_by_feature'] = mbf_days(n_extra)
    return o

def et_factory(seed):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy='median'),
                         ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=2))
def cb_factory(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3, thread_count=2,
                              random_seed=seed, verbose=0, allow_writing_files=False)
def cb_hook(kw, cat_cols, names, Xtr):
    if cat_cols: kw['cat_features'] = list(cat_cols)
    return kw
cat = pipe.lgb_cat_hook

jobs = [
 # C: exact date as LightGBM categorical, added to F2 (reference: A_L_mb255_mdb1 and L)
 ('C_L_cat_cs1_mdpg5',   dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=cat_mbf, cat_smooth=1, min_data_per_group=5),  blocks=[pipe.block_date_cat], fit_hook=cat)),
 ('C_L_cat_cs10_mdpg20', dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=cat_mbf, cat_smooth=10, min_data_per_group=20), blocks=[pipe.block_date_cat], fit_hook=cat)),
 # H: ExtraTrees OOF for blending
 ('H_ET', dict(model_fn=et_factory)),
 # G: extra estimator seeds of L+
 ('G_Lp_seed1', dict(model_fn=pipe.lgb_factory(), blocks=Lp_blocks, model_seeds=(1,))),
 ('G_Lp_seed2', dict(model_fn=pipe.lgb_factory(), blocks=Lp_blocks, model_seeds=(2,))),
 # A6: best histogram setting on L+
 ('A_Lp_best', dict(model_fn=pipe.lgb_factory(**best_over(2)), blocks=Lp_blocks)),
 # B with the best histogram setting
 ('B_L_mcs5_bestbins', dict(model_fn=pipe.lgb_factory(min_child_samples=5, **best_over(0)))),
 ('B_L_mcs3_bestbins', dict(model_fn=pipe.lgb_factory(min_child_samples=3, **best_over(0)))),
 # noise calibration: L+ with a different inner cross-fit seed
 ('L+_inner1', dict(model_fn=pipe.lgb_factory(), blocks=Lp_blocks, note='INNER_SEED=1')),
 # D: CatBoost with / without the raw date string categorical
 ('D_CB_F2',     dict(model_fn=cb_factory)),
 ('D_CB_F2_cat', dict(model_fn=cb_factory, blocks=[pipe.block_date_str], fit_hook=cb_hook)),
 # C: remaining grid, and replace-days variant
 ('C_L_cat_cs10_mdpg5',  dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=cat_mbf, cat_smooth=10, min_data_per_group=5), blocks=[pipe.block_date_cat], fit_hook=cat)),
 ('C_L_cat_cs1_mdpg20',  dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=cat_mbf, cat_smooth=1, min_data_per_group=20), blocks=[pipe.block_date_cat], fit_hook=cat)),
 ('C_L_catREPL_cs1_mdpg5', dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=[255]*(len(F2)-1)+[4095], cat_smooth=1, min_data_per_group=5), blocks=[pipe.block_date_cat], fit_hook=cat, base_cols=[c for c in F2 if c != 'days'])),
 ('C_Lp_cat_cs1_mdpg5', dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=[255]*(len(F2)+2)+[4095], cat_smooth=1, min_data_per_group=5), blocks=Lp_blocks+[pipe.block_date_cat], fit_hook=cat)),
 # G: seeds 3, 4
 ('G_Lp_seed3', dict(model_fn=pipe.lgb_factory(), blocks=Lp_blocks, model_seeds=(3,))),
 ('G_Lp_seed4', dict(model_fn=pipe.lgb_factory(), blocks=Lp_blocks, model_seeds=(4,))),
]
for name, kw in jobs:
    try:
        if name == 'L+_inner1':
            pipe.INNER_SEED = 1
        pipe.evaluate(name, **kw)
    except Exception:
        print('FAILED', name); traceback.print_exc()
    finally:
        pipe.INNER_SEED = 0
print('Q2 DONE', time.strftime('%H:%M:%S'), flush=True)
