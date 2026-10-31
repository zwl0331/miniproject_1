import pandas as pd, numpy as np
P='/Users/zwl0331/Desktop/Mini_project_1/'
tr=pd.read_csv(P+'data/train.csv'); te=pd.read_csv(P+'data/test.csv'); sp=pd.read_csv(P+'artifacts/splits/production_v1.csv')
print(tr.shape, te.shape, sp.shape)
print(tr.isna().sum().to_string())
print(tr.iloc[425])
tr['y']=(tr.Popularity_Type=='High').astype(int)
print('High rate', tr.y.mean())
aud=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
print(tr[aud].describe().T.to_string())
print(te[aud].describe().T.to_string())
# date formats
def fmt(s):
    s=str(s); return 'YMD' if len(s)==10 else ('Y' if len(s)==4 else ('YM' if len(s)==7 else 'other'))
tr['fmt']=tr.track_album_release_date.map(fmt); te['fmt']=te.track_album_release_date.map(fmt)
print(tr.fmt.value_counts(), te.fmt.value_counts())
print(tr.groupby('fmt').y.agg(['mean','count']))
# year
tr['year']=tr.track_album_release_date.str[:4].astype(int); te['year']=te.track_album_release_date.str[:4].astype(int)
print(tr.groupby(pd.cut(tr.year,[1950,1990,2000,2010,2015,2018,2020,2021,2022,2023,2024,2025])).y.agg(['mean','count']))
print(te.groupby(pd.cut(te.year,[1950,1990,2000,2010,2015,2018,2020,2021,2022,2023,2024,2025])).size())
# group structure
print('groups', sp.primary_group.nunique(), 'eligible', sp.eligible.sum())
g=sp.groupby('primary_group').size(); print(g.value_counts())
# duplicates with conflicting labels
tr['grp']=sp.primary_group.values
agg=tr[tr.eligible if 'eligible' in tr else sp.eligible.values==1].groupby('grp').y.agg(['mean','count'])
multi=agg[agg['count']>1]
print('multi groups', len(multi), 'conflicting', ((multi['mean']>0)&(multi['mean']<1)).sum())
print(multi['mean'].value_counts())
