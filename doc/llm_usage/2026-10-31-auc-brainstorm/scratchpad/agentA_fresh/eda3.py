import pandas as pd, numpy as np
P='/Users/zwl0331/Desktop/Mini_project_1/'
tr=pd.read_csv(P+'data/train.csv'); te=pd.read_csv(P+'data/test.csv'); sp=pd.read_csv(P+'artifacts/splits/production_v1.csv')
tr['y']=(tr.Popularity_Type=='High').astype(int); tr['grp']=sp.primary_group.values; tr['elig']=sp.eligible.values
aud=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
tr=tr[tr.elig==1].copy()
d='track_album_release_date'
# dedupe to one row per group for date analysis
g=tr.groupby('grp').agg(y=('y','mean'),date=(d,'first'),n=('y','size')).reset_index(); g['y']=(g.y>0.5).astype(int)
g['year']=g.date.str[:4].astype(int)
dc=g.groupby('date').agg(n=('y','size'),m=('y','mean'))
print('distinct dates among groups:', len(dc), 'dates with >=2 groups:', (dc.n>=2).sum(), 'groups on shared dates:', dc[dc.n>=2].n.sum())
# label coherence on shared dates: leave-one-out date mean vs own label, compared with leave-one-out year mean
g['date_n']=g.date.map(dc.n); g['date_sum']=g.date.map(dc.n*dc.m)
g['loo_date']=(g.date_sum-g.y)/(g.date_n-1)
yc=g.groupby('year').agg(n=('y','size'),m=('y','mean'))
g['year_n']=g.year.map(yc.n); g['loo_year']=(g.year.map(yc.n*yc.m)-g.y)/(g.year_n-1)
sh=g[g.date_n>=2]
from sklearn.metrics import roc_auc_score
print('shared-date groups', len(sh), 'AUC loo_date', roc_auc_score(sh.y,sh.loo_date), 'AUC loo_year', roc_auc_score(sh.y,sh.loo_year))
# per-year-bucket
for lo,hi in [(1950,2000),(2000,2015),(2015,2021),(2021,2025)]:
    s=sh[(sh.year>=lo)&(sh.year<hi)]
    if s.y.nunique()>1: print(lo,hi,len(s), 'loo_date',round(roc_auc_score(s.y,s.loo_date),3),'loo_year',round(roc_auc_score(s.y,s.loo_year),3))
# test date overlap
tdc=te[d].value_counts()
print('test rows whose date appears in train:', te[d].isin(dc.index).sum(), 'of', len(te))
print('test rows whose date appears in train >=2 times:', te[d].isin(dc[dc.n>=2].index).sum())
# weekday structure
tr['dt']=pd.to_datetime(tr[d],errors='coerce'); tr['wd']=tr.dt.dt.weekday; tr['year']=tr[d].str[:4].astype(int)
print(tr[tr.year>=2016].groupby('wd').y.agg(['mean','count']))
print(tr[(tr.year<2015)&(tr.year>=2000)].groupby('wd').y.agg(['mean','count']))
# day==1 (month precision masquerading) 
tr['day']=tr.dt.dt.day
print('day==1 rate', tr[tr.year>=2000].groupby(tr.day==1).y.agg(['mean','count']))
print('jan 1', tr.groupby((tr.dt.dt.month==1)&(tr.day==1)).y.agg(['mean','count']))
# audio by label
print(tr.groupby('y')[aud].mean().T)
# how many rows per date in test vs train by year
print('dates in test with multiple test rows:', (tdc>=2).sum(), tdc[tdc>=2].sum())
