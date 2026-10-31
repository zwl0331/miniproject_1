import pipe, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
d = pipe.load(); y = d['y']
tr_idx, va_idx = pipe.fold_indices()[(155,0)]
Xtr, Xva, cat_cols, names = pipe.build_fold(tr_idx, va_idx, pipe.F2_COLS, [pipe.block_date_cat])
print('cat cols', cat_cols, 'n distinct train dates', len(np.unique(Xtr[:,cat_cols[0]])), 'unseen va', (Xva[:,cat_cols[0]]==len(np.unique(Xtr[:,cat_cols[0]]))).mean())
import pandas as pd
cnt = pd.Series(Xtr[:,cat_cols[0]]).value_counts()
print('date count distribution in training fold:', cnt.value_counts().sort_index().head(10).to_dict())
for mdb, mb in [(3,255),(1,255),(1,4095)]:
    p = dict(pipe.LGB_BASE); p.update(n_estimators=100, min_data_in_bin=mdb, max_bin=mb, min_data_per_group=5, cat_smooth=10)
    m = lgb.LGBMClassifier(**p); m.fit(Xtr, y[tr_idx], categorical_feature=cat_cols)
    dump = m.booster_.dump_model()
    fn = dump["feature_names"][cat_cols[0]]; fi = dump["feature_infos"][fn]
    ncat = len(fi.strip('[]').split(':')) if isinstance(fi,str) else fi
    imp = m.booster_.feature_importance('split')
    print(f'mdb={mdb} mb={mb}: cat bins listed={ncat}  splits on cat={imp[cat_cols[0]]} of {imp.sum()}  splits on days={imp[names.index("days")]}')
    print('   feature_infos sample:', str(fi)[:120])
