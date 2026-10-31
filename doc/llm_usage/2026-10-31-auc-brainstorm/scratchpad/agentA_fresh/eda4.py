import sys; sys.path.insert(0,'.')
from pipe import *
from sklearn.preprocessing import StandardScaler
tr,te=load()
g=tr.groupby('grp').agg(y=('y','mean'),date=(D,'first'),**{c:(c,'first') for c in AUD},year=('year','first')).reset_index()
g['yb']=(g.y>0.5).astype(int)
A=StandardScaler().fit_transform(g[AUD].values)
# pairs sharing a date
pairs=[]
for d,idx in g.groupby('date').indices.items():
    if len(idx)<2: continue
    for i in range(len(idx)):
        for j in range(i+1,len(idx)):
            a,b=idx[i],idx[j]; pairs.append((np.linalg.norm(A[a]-A[b]), abs(A[a][AUD.index('loudness')]-A[b][AUD.index('loudness')]), g.yb[a]==g.yb[b], g.year[a]))
P=pd.DataFrame(pairs,columns=['dist','dloud','agree','year'])
print('same-date pairs:',len(P)); P['q']=pd.qcut(P.dist,5)
print(P.groupby('q',observed=True).agree.agg(['mean','count']))
# random pairs same year as reference
rng=np.random.default_rng(0); rp=[]
for y_,idx in g.groupby('year').indices.items():
    if len(idx)<2: continue
    for _ in range(min(200,len(idx)*3)):
        a,b=rng.choice(idx,2,replace=False); rp.append((np.linalg.norm(A[a]-A[b]), g.yb[a]==g.yb[b]))
R=pd.DataFrame(rp,columns=['dist','agree']); R['q']=pd.qcut(R.dist,5)
print('random same-year pairs agreement by distance quintile:'); print(R.groupby('q',observed=True).agree.agg(['mean','count']))
print('base agreement same-year random:', R.agree.mean(), ' same-date:', P.agree.mean())
