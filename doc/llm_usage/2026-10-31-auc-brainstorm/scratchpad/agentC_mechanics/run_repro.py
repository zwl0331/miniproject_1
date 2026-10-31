import pipe, numpy as np, time
d = pipe.load()
X = d['X']
print(X.shape, X.isna().sum().to_dict())
print(X.describe().T[['min','max','count']])
# sanity on blocks
tr_idx, va_idx = pipe.fold_indices()[(155,0)]
a, b = pipe.block_date_count(tr_idx, va_idx); print('date_count', a.shape, np.nanmin(a), np.nanmax(a), b.min(), b.max())
te = pipe.make_block_date_te(3.0); a, b = te(tr_idx, va_idx); print('te3', a.shape, np.isnan(a).sum(), a.min(), a.max(), b.min(), b.max())
t0=time.time()
L = pipe.evaluate('L', pipe.lgb_factory())
Lp = pipe.evaluate('L+', pipe.lgb_factory(), blocks=[pipe.block_date_count, pipe.make_block_date_te(3.0)])
