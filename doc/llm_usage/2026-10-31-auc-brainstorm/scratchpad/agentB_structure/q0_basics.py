from common import *
tr,te,sp=load()
print(tr.shape,te.shape,sp.shape)
print(tr.isna().sum()[tr.isna().sum()>0])
print('row 425:',tr.iloc[425][['uri','track_album_release_date','Popularity_Type','duration_ms']].to_dict())
print(sp.eligible.value_counts(), sp.primary_group.nunique())
print('date_len train',tr.date_len.value_counts().to_dict(),'test',te.date_len.value_counts().to_dict())
print('High rate',tr.y.mean())
print(tr.groupby(pd.cut(tr.year,[0,1989,1999,2009,2019,2030]))['y'].agg(['mean','size']))
# check sp alignment
print('sp train_row_id == index:', (sp.train_row_id.values==np.arange(len(sp))).all())
# fold sizes
for s in ['155','2026','1029']:
    print(s, sp[f'primary_fold_{s}'].value_counts().sort_index().to_dict())
# groups: how do groups relate to uri
g=sp.groupby('primary_group').apply(lambda d: tr.loc[d.index,'uri'].nunique())
print('groups with >1 uri:', (g>1).sum(), 'uri with >1 group:', tr.assign(g=sp.primary_group).groupby('uri')['g'].nunique().gt(1).sum())
print('ineligible rows:', sp[sp.eligible==0].index.tolist()[:20], len(sp[sp.eligible==0]))
