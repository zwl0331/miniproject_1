import pipe, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
d = pipe.load(); y = d['y']; dates = d['dates']
fi = pipe.fold_indices()
seen, nn_cov, aucs = [], [], {}
for (s,k),(tr,va) in fi.items():
    tr_d = set(dates[tr]); seen.append(np.mean([x in tr_d for x in dates[va]]))
    if s == 155:
        trb, vab = pipe.block_date_nn(tr, va)
        nn_cov.append(np.isfinite(vab[:,0]).mean())
        te = pipe.make_block_date_te(3.0); ttr, tva = te(tr, va)
        cb_tr, cb_va = pipe.block_date_count(tr, va)
        m = np.isfinite(vab[:,0])
        aucs.setdefault('te3 alone (va rows)', []).append(roc_auc_score(y[va], tva[:,0]))
        aucs.setdefault('date_count alone (va rows)', []).append(roc_auc_score(y[va], cb_va[:,0]))
        aucs.setdefault('nn_label alone (va rows with neighbour)', []).append(roc_auc_score(y[va][m], vab[m,0]))
        aucs.setdefault('kernel mean alone (va rows with neighbour)', []).append(roc_auc_score(y[va][m], vab[m,2]))
        aucs.setdefault('te3 on the same subset', []).append(roc_auc_score(y[va][m], tva[m,0]))
        if k == 0:
            print('fold 155/0: train NN-feature NaN rate', np.isnan(trb[:,0]).mean().round(3), 'val NaN rate', np.isnan(vab[:,0]).mean().round(3))
            print('  h (median train NN dist) ~', np.nanmedian(trb[:,1]).round(3), ' val NN dist quantiles', np.nanpercentile(vab[:,1],[10,50,90]).round(2))
            print('  n_same_date val quantiles', np.nanpercentile(vab[:,3],[50,90,99]))
            print('  te3 train range', ttr.min().round(3), ttr.max().round(3), ' te3 val range', tva.min().round(3), tva.max().round(3))
print('fraction of held-out rows whose exact date occurs in the training fold: mean %.3f (min %.3f max %.3f)' % (np.mean(seen), min(seen), max(seen)))
print('fraction of held-out rows with a same-date neighbour (seed 155 folds):', np.round(nn_cov,3))
for k,v in aucs.items(): print(f'  AUC {k}: {np.mean(v):.4f}')
# label agreement within exact date across groups
df = pd.DataFrame({'d':dates,'y':y,'g':d['groups']})
cnt = df.groupby('d').g.nunique()
print('dates with >=2 distinct groups:', (cnt>=2).sum(), 'of', len(cnt), '; rows on such dates:', df.d.map(cnt).ge(2).sum())
# year-level High rate for context
yr = pd.Series(d['X']['year'].values)
print(pd.DataFrame({'yr':(yr//10*10).astype(int),'y':y}).groupby('yr').y.agg(['mean','count']).T.round(2).to_string())
