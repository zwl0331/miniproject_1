import pandas as pd, numpy as np
P='/Users/zwl0331/Desktop/Mini_project_1/'
tr=pd.read_csv(P+'data/train.csv'); te=pd.read_csv(P+'data/test.csv'); sp=pd.read_csv(P+'artifacts/splits/production_v1.csv')
tr['y']=(tr.Popularity_Type=='High').astype(int)
aud=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
tr['grp']=sp.primary_group.values; tr['elig']=sp.eligible.values
def hk(df,cols): return pd.util.hash_pandas_object(df[cols], index=False).values
key=aud+['track_album_release_date']
tr['k']=hk(tr,key); te['k']=hk(te,key); tr['ka']=hk(tr,aud); te['ka']=hk(te,aud)
print('unique exact keys (train):', tr.k.nunique(), 'groups:', tr.grp.nunique())
x=tr.groupby('grp').k.nunique(); print('groups with >1 distinct exact key:', (x>1).sum())
y=tr.groupby('k').grp.nunique(); print('exact keys spanning >1 group:', (y>1).sum())
d=tr[tr.grp.duplicated(keep=False)]
print('dup rows', len(d), 'uri nunique among dup rows', d.uri.nunique(), 'groups', d.grp.nunique())
print(d.sort_values('grp')[['grp','uri','track_album_release_date','danceability','tempo','y']].head(12).to_string())
print('audio-only unique', tr.ka.nunique())
ga=tr.groupby('ka').agg(n=('y','size'),ndate=('track_album_release_date','nunique'))
print('audio-key dup groups with >1 date:', (ga.ndate>1).sum())
trk=set(tr.k); trka=set(tr.ka)
print('test rows with exact (audio+date) match in train:', te.k.isin(trk).sum(), ' audio-only:', te.ka.isin(trka).sum())
print('test rows duplicated within test:', te.k.duplicated(keep=False).sum())
m=te[te.k.isin(trk)]
lab=tr.groupby('k').y.mean()
print('mean train label of matched test rows:', lab.loc[m.k].mean(), 'n', len(m))
allk=pd.concat([tr.k,te.k]); cnt=allk.value_counts()
tr['cnt']=tr.k.map(cnt); te['cnt']=te.k.map(cnt)
print(tr[tr.elig==1].groupby('cnt').y.agg(['mean','count']))
print(te.cnt.value_counts())
tr['year']=tr.track_album_release_date.str[:4].astype(int)
s=tr[(tr.elig==1)]
print(s.groupby([pd.cut(s.year,[1950,2000,2015,2020,2025]), s.cnt>1]).y.agg(['mean','count']))
# near-duplicates: same date + same duration_ms but differing other features?
tr['kd']=hk(tr,['duration_ms','track_album_release_date'])
gd=tr.groupby('kd').agg(n=('y','size'),nk=('k','nunique'))
print('date+duration groups with >1 distinct exact key:', (gd.nk>1).sum(), 'rows', gd[gd.nk>1].n.sum())
# how many exact-dup rows are in different folds (sanity)
for f in ['primary_fold_155','primary_fold_2026','primary_fold_1029']:
    tr[f]=sp[f].values
    print(f, 'keys spanning >1 fold:', (tr.groupby('k')[f].nunique()>1).sum())
