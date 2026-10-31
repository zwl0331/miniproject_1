from common import *
from scipy import stats
tr,te,sp=load()
tr=tr.join(sp[['eligible','primary_group']]); el=tr[tr.eligible==1].reset_index(drop=True)
Ztr,Zte=zfeat(el,te); Ztr=Ztr.values; Zte=Zte.values
ds_tr=el.date_str.values; ds_te=te.date_str.values
print('=== Q2: test rows vs FULL train (deployment)')
bydate={}
for i,d in enumerate(ds_tr): bydate.setdefault(d,[]).append(i)
nn=np.full(len(te),np.nan); cnt=np.zeros(len(te)); nnlab=np.full(len(te),np.nan)
for i in range(len(te)):
    c=bydate.get(ds_te[i])
    if not c: continue
    c=np.array(c); d=np.linalg.norm(Ztr[c]-Zte[i],axis=1); cnt[i]=len(c); nn[i]=d.min(); nnlab[i]=el.y.values[c[np.argmin(d)]]
full=te.date_len.values==10
print(f'test rows with same-date train row: {np.isfinite(nn).mean():.3f} (full-date rows only: {np.isfinite(nn[full]).mean():.3f})')
for t in [0.5,1.0,1.5,2.0,2.34,3.02]:
    print(f'  nn_dist < {t}: {(nn<t).mean():.3f}  (n={(nn<t).sum()})')
print('  mean count of same-date train rows (when >0):',cnt[cnt>0].mean().round(2),' median',np.median(cnt[cnt>0]))
print('  nearest same-date label mean among test rows with a mate:',np.nanmean(nnlab).round(3))
print('CV reference (val rows vs 80% train): with mate 0.693, nn<1: 0.020, nn<2.34: 0.251  [from q1_probe]')
# within-train full-data reference: each train row vs other-group train rows (100% of other rows)
nn_tr=np.full(len(el),np.nan)
g=el.primary_group.values
for i in range(len(el)):
    c=bydate.get(ds_tr[i]); 
    if not c: continue
    c=np.array(c); c=c[g[c]!=g[i]]
    if len(c)==0: continue
    nn_tr[i]=np.linalg.norm(Ztr[c]-Ztr[i],axis=1).min()
print(f'train rows vs all other-group train rows: with mate {np.isfinite(nn_tr).mean():.3f}, nn<1 {(nn_tr<1).mean():.3f}, nn<2.34 {(nn_tr<2.34).mean():.3f}')
# test dates not in train: how many, by year
unseen=~np.isin(ds_te,list(bydate.keys()))
print('test date strings unseen in train:',unseen.sum(),' by date_len:',pd.Series(te.date_len.values[unseen]).value_counts().to_dict())
print('test year distribution vs train:',pd.Series(pd.cut(te.year,[0,1989,1999,2009,2019,2030])).value_counts(normalize=True).sort_index().round(3).to_dict())
print('                                ',pd.Series(pd.cut(el.year,[0,1989,1999,2009,2019,2030])).value_counts(normalize=True).sort_index().round(3).to_dict())

print('\n=== Q4: within-test structure')
vc=te.uri.value_counts(); rep=vc[vc>1]
print('repeated uri in test:',len(rep),'rows involved:',rep.sum(),' multiplicities:',rep.value_counts().to_dict())
# are repeated-uri rows identical in audio?
for u in rep.index[:11]:
    sub=te[te.uri==u]; same=(sub[AUDIO].nunique()<=1).all(); print(f'  {u[-8:]}: n={len(sub)} identical_audio={same} dates={sub.date_str.unique().tolist()}')
dvc=te.date_str.value_counts()
print('test rows sharing an exact date with another test row:',(te.date_str.map(dvc)>1).sum(),' distinct such dates:',(dvc>1).sum(),' max per date:',dvc.max())
# same-date audio-near clusters inside test (different uri)
import itertools
pairs=[]
for d,grp in te.groupby('date_str'):
    if len(grp)<2 or len(d)!=10: continue
    ii=grp.index.values
    for a,b in itertools.combinations(ii,2):
        if te.uri[a]!=te.uri[b]: pairs.append((a,b,np.linalg.norm(Zte[a]-Zte[b])))
P=pd.DataFrame(pairs,columns=['a','b','d'])
print('same-date different-uri test pairs:',len(P),' d<1:',(P.d<1).sum(),' d<2.34:',(P.d<2.34).sum(),' d<3.02:',(P.d<3.02).sum())
near=P[P.d<2.34]; print('test rows in a same-date pair with d<2.34:',len(set(near.a)|set(near.b)))
# test-test same-date pairs where neither has a train mate (transductive-only information)
has_tr=np.isfinite(nn)
print('test rows with a same-date TEST mate but NO same-date train row:',((te.date_str.map(dvc)>1)&~has_tr).sum())
# also: test rows whose date is unseen in train but share date with other test rows: which dates
tmp=te[(te.date_str.map(dvc)>1)&~has_tr].date_str.value_counts(); print('  those dates:',tmp.to_dict())
# audio-only near duplicates test->train (any date) : exact signature
sig_tr=el[AUDIO].round(6).astype(str).agg('|'.join,axis=1); sig_te=te[AUDIO].round(6).astype(str).agg('|'.join,axis=1)
print('test rows whose exact audio signature appears in train:',sig_te.isin(set(sig_tr)).sum())
from scipy.spatial import cKDTree
tree=cKDTree(Ztr); dd,ii=tree.query(Zte,k=1)
print('test->train nearest audio distance (any date) quantiles:',np.round(np.quantile(dd,[0.01,0.05,0.1,0.25,0.5]),3),' n with d<0.3:',(dd<0.3).sum())
same_year=(el.year.values[ii]==te.year.values); print('  of test rows with d<0.3 to a train row, fraction with same date string:',(ds_tr[ii][dd<0.3]==ds_te[dd<0.3]).mean().round(3),' same year:',same_year[dd<0.3].mean().round(3))

print('\n=== Q5: test ID gaps')
ids=te.ID.values; allid=np.arange(2,968); missing=np.setdiff1d(allid,ids)
print('IDs present',len(ids),'range',ids.min(),ids.max(),'missing',len(missing))
print('missing IDs:',missing.tolist())
# runs
runs=np.split(missing,np.where(np.diff(missing)!=1)[0]+1); rl=pd.Series([len(r) for r in runs]).value_counts().sort_index()
print('runs of consecutive missing IDs, length counts:',rl.to_dict())
# gaps between consecutive missing
gaps=np.diff(missing); print('gaps between consecutive missing IDs: mean',gaps.mean().round(2),'sd',gaps.std().round(2),'min',gaps.min(),'max',gaps.max())
# null: 103 uniformly random missing out of 966 -> expected number of runs and gap sd
rng=np.random.default_rng(0); sim_runs=[];sim_sd=[]
for _ in range(2000):
    m=np.sort(rng.choice(allid,len(missing),replace=False)); sim_runs.append(len(np.split(m,np.where(np.diff(m)!=1)[0]+1))); sim_sd.append(np.diff(m).std())
print(f'  observed runs {len(runs)} vs null {np.mean(sim_runs):.1f}+-{np.std(sim_runs):.1f}; observed gap sd {gaps.std():.2f} vs null {np.mean(sim_sd):.2f}+-{np.std(sim_sd):.2f}')
# periodicity: missing mod k
for k in [2,3,5,7,8,9,10,12]:
    obs=np.bincount(missing%k,minlength=k); chi=stats.chisquare(obs).pvalue; print(f'  missing mod {k}: {obs.tolist()} chi2 p={chi:.3f}')
# density of missing along the ID axis
bins=np.linspace(2,968,11); h=np.histogram(missing,bins)[0]; print('  missing per decile of ID range:',h.tolist(),' chi2 p=',round(stats.chisquare(h).pvalue,3))
# relation to neighbours' features: is a row adjacent to a missing ID different?
adj=np.isin(ids-1,missing)|np.isin(ids+1,missing)
print('test rows adjacent to a missing ID:',adj.sum())
for c in ['year','duration_ms','loudness','energy','date_len']:
    a=te[c].values[adj]; b=te[c].values[~adj]; p=stats.mannwhitneyu(a,b).pvalue; print(f'   {c}: adj mean {np.nanmean(a):.3f} vs other {np.nanmean(b):.3f}  MWU p={p:.3f}')
# does ID order relate to anything in test
for c in ['year','duration_ms','loudness','energy','danceability']:
    r=stats.spearmanr(ids,te[c].values,nan_policy='omit'); print(f'  spearman(ID,{c}) = {r.statistic:+.3f} p={r.pvalue:.3f}')
# hypothesis: missing IDs = test rows removed for duplicates with train? sizes: train 3864 + test 863 + 103 missing = 4830; 
print('train+test+missing =',3864+863+103, '; train/(train+test) =',round(3864/(3864+863),4),' with missing: ',round(3864/4830,4))
# are there any test uris with identical audio in train but different date -> maybe such were removed
np.save('te_nn_dist.npy',nn)
