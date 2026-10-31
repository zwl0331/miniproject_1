from common import *
import itertools
rng=np.random.default_rng(0)
tr,te,sp=load()
tr=tr.join(sp[['eligible','primary_group']])
el=tr[tr.eligible==1].copy()
Z=zfeat(el)
Zv=Z.values
y=el.y.values; g=el.primary_group.values; ds=el.date_str.values; yr=el.year.values; dl=el.date_len.values
idx=np.arange(len(el))
# ---- same-date pairs (full dates only), different group
def pairs_same(key_arr, mask):
    out=[]
    df=pd.DataFrame({'k':key_arr,'i':idx})[mask]
    for k,grp in df.groupby('k'):
        ii=grp.i.values
        if len(ii)<2: continue
        for a,b in itertools.combinations(ii,2):
            if g[a]!=g[b]: out.append((a,b))
    return np.array(out)
P=pairs_same(ds, dl==10)
d_same=np.linalg.norm(Zv[P[:,0]]-Zv[P[:,1]],axis=1)
agree=(y[P[:,0]]==y[P[:,1]]).astype(float)
print('same full-date different-group pairs:',len(P),'rows involved:',len(np.unique(P)), 'agreement:',agree.mean().round(4))
# null: same year, different date string, different group; subsample up to 300k pairs by sampling
def null_pairs(nmax=400000):
    out=[]
    df=pd.DataFrame({'yr':yr,'i':idx})[dl==10]
    for k,grp in df.groupby('yr'):
        ii=grp.i.values
        if len(ii)<2: continue
        pp=np.array(list(itertools.combinations(ii,2)))
        pp=pp[(ds[pp[:,0]]!=ds[pp[:,1]])&(g[pp[:,0]]!=g[pp[:,1]])]
        out.append(pp)
    return np.vstack(out)
N=null_pairs()
d_null=np.linalg.norm(Zv[N[:,0]]-Zv[N[:,1]],axis=1)
agree_null=(y[N[:,0]]==y[N[:,1]]).astype(float)
print('same-year different-date pairs:',len(N),'agreement:',agree_null.mean().round(4))
# matched-by-year reweighting: weight null pairs so the year distribution matches same-date pairs
yp=pd.Series(yr[P[:,0]]).value_counts(normalize=True); yn=pd.Series(yr[N[:,0]]).value_counts(normalize=True)
w=np.array([yp.get(v,0)/yn.get(v,1) for v in yr[N[:,0]]])
print('year-matched null agreement:',np.average(agree_null,weights=w).round(4))
# expected agreement from year label rates
# positive-pair / negative-pair breakdown
bothH=((y[P[:,0]]==1)&(y[P[:,1]]==1)).mean(); bothL=((y[P[:,0]]==0)&(y[P[:,1]]==0)).mean()
bothHn=np.average((y[N[:,0]]==1)&(y[N[:,1]]==1),weights=w); bothLn=np.average((y[N[:,0]]==0)&(y[N[:,1]]==0),weights=w)
print(f'same-date: bothHigh {bothH:.4f} bothLow {bothL:.4f} | matched null: bothHigh {bothHn:.4f} bothLow {bothLn:.4f}')
# agreement vs distance quantile, thresholds from same-date distance quantiles
qs=[0,0.1,0.25,0.5,0.75,0.9,1.0]
th=np.quantile(d_same,qs); th[-1]=np.inf; th[0]=-1
print('distance thresholds (same-date quantiles):',np.round(th,3))
rows=[]
for lo,hi,ql,qh in zip(th[:-1],th[1:],qs[:-1],qs[1:]):
    m=(d_same>lo)&(d_same<=hi); mn=(d_null>lo)&(d_null<=hi)
    rows.append(dict(q=f'{ql}-{qh}',d_lo=round(lo,2),d_hi=round(hi,2),n_same=m.sum(),agree_same=agree.mean() if m.sum()==0 else agree[m].mean(),
                     n_null=mn.sum(),agree_null_matched=np.average(agree_null[mn],weights=w[mn]) if mn.sum()>0 else np.nan,
                     agree_null_raw=agree_null[mn].mean() if mn.sum()>0 else np.nan))
T=pd.DataFrame(rows); print(T.round(4).to_string())
# finer: nearest neighbour analysis: for each row with same-date mates, label agreement with nearest mate vs nearest same-year different-date row
# distribution of overall nearest-neighbour distances among all rows (for reference)
from scipy.spatial import cKDTree
tree=cKDTree(Zv); dd,ii=tree.query(Zv,k=2)
print('overall NN distance quantiles (any row):',np.round(np.quantile(dd[:,1],[0.1,0.25,0.5,0.75,0.9]),3))
# per-row: nearest same-date mate
best={}
for a,b,d in zip(P[:,0],P[:,1],d_same):
    for u,v in ((a,b),(b,a)):
        if u not in best or d<best[u][1]: best[u]=(v,d)
rows=np.array(list(best.keys())); nnv=np.array([best[r][0] for r in rows]); nnd=np.array([best[r][1] for r in rows])
print('rows with a same-date mate:',len(rows),'of',len(el))
ag=(y[rows]==y[nnv]).astype(float)
for lo,hi in [(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,2.5),(2.5,3),(3,4),(4,99)]:
    m=(nnd>=lo)&(nnd<hi)
    if m.sum(): print(f'  nearest mate dist [{lo},{hi}): n={m.sum():4d} agree={ag[m].mean():.3f}  P(High|mate High)={y[rows][m][y[nnv][m]==1].mean() if (y[nnv][m]==1).any() else np.nan:.3f} P(High|mate Low)={y[rows][m][y[nnv][m]==0].mean() if (y[nnv][m]==0).any() else np.nan:.3f}')
# AUC of nearest mate label alone among rows with a mate
from sklearn.metrics import roc_auc_score
print('AUC(nearest same-date mate label) among rows with mate:',round(roc_auc_score(y[rows],y[nnv]),4),' base rate among them',y[rows].mean().round(3))
# same for same-year nearest (null): nearest different-date, same-year, different-group row
nn_y=[]
for r in rows:
    m=(yr==yr[r])&(ds!=ds[r])&(g!=g[r])&(dl==10)
    cand=np.where(m)[0]
    if len(cand)==0: nn_y.append(np.nan); continue
    dd_=np.linalg.norm(Zv[cand]-Zv[r],axis=1); nn_y.append(y[cand[np.argmin(dd_)]])
nn_y=np.array(nn_y); ok=~np.isnan(nn_y)
print('AUC(nearest same-YEAR other-date label) same rows:',round(roc_auc_score(y[rows][ok],nn_y[ok]),4))
# mean same-date mate label (date TE without self group) AUC
mean_lab=np.array([np.mean([y[b] for b in np.r_[P[P[:,0]==r,1],P[P[:,1]==r,0]]]) for r in rows])
print('AUC(mean same-date mates label):',round(roc_auc_score(y[rows],mean_lab),4))
# YYYY-only dates: pairs sharing same YYYY string
P4=pairs_same(ds, dl==4)
if len(P4): print('YYYY same-string pairs:',len(P4),'agreement',(y[P4[:,0]]==y[P4[:,1]]).mean().round(3))
np.save('same_date_pairs.npy',P); np.save('same_date_dist.npy',d_same)
