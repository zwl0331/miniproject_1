from common import *
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(7)
tr,te,sp=load()
tr=tr.join(sp[['eligible','primary_group','primary_fold_155','primary_fold_2026','primary_fold_1029']])
el=tr[tr.eligible==1].reset_index(drop=True); y=el.y.values; g=el.primary_group.values; ds=el.date_str.values
Zf=zfeat(el).values
O=np.load('oof_probe.npz')
uvc=el.uri.value_counts(); nc=el.uri.map(uvc).values
print('=== duplicate rows: label rate by n_copies:',{k:(round(y[nc==k].mean(),3),int((nc==k).sum())) for k in [1,2,3]})
print('=== CV population vs test population: AUC of OOF model A by seed')
for s in ['155','2026','1029']:
    p=O[f'A_base_{s}']; print(f'  seed {s}: all rows {roc_auc_score(y,p):.4f}  singleton-uri rows {roc_auc_score(y[nc==1],p[nc==1]):.4f}  duplicate rows {roc_auc_score(y[nc>1],p[nc>1]):.4f}  High rate dup {y[nc>1].mean():.3f}; model mean score on dup rows {p[nc>1].mean():.3f} vs singletons {p[nc==1].mean():.3f}')
print('=== emulate the test population: singleton rows + 22/863 share of duplicate pairs; boost duplicate rows to the top')
gains=[];base=[];rank_dup=[]
pairs=[grp for grp in el.groupby('uri').indices.values() if len(grp)==2]
for s in ['155','2026','1029']:
    p=O[f'A_base_{s}']
    for _ in range(300):
        sing=np.where(nc==1)[0]; n_pairs=int(round(len(sing)*22/841/2))
        chosen=np.concatenate([pairs[i] for i in rng.choice(len(pairs),n_pairs,replace=False)])
        pop=np.concatenate([sing,chosen]); yy=y[pop]; pp=p[pop].copy()
        a0=roc_auc_score(yy,pp); pp2=pp.copy(); pp2[len(sing):]+=1.0; a1=roc_auc_score(yy,pp2)
        base.append(a0); gains.append(a1-a0)
        rank_dup.append(np.mean([(pp<v).mean() for v in pp[len(sing):]]))
print(f'  emulated test-like AUC without boost {np.mean(base):.4f}; gain from n_copies boost {np.mean(gains):+.4f} (sd over draws {np.std(gains):.4f}); mean percentile of duplicate rows under the model: {np.mean(rank_dup):.3f}')
# also what the gain looks like in plain CV (all validation rows, 11% dups)
for s in ['155']:
    p=O[f'A_base_{s}']; p2=p+(nc>1); print(f'  same boost on the full CV population (11% dups): {roc_auc_score(y,p):.4f} -> {roc_auc_score(y,p2):.4f}')
print('\n=== near-duplicate (different group, same date, audio distance < t) override on OOF predictions, 15 folds')
def nnfeat(trm):
    n=len(el); lab=np.full(n,np.nan); dist=np.full(n,np.nan); bydate={}
    for i in np.where(trm)[0]: bydate.setdefault(ds[i],[]).append(i)
    for i in range(n):
        c=bydate.get(ds[i]); 
        if not c: continue
        c=np.array(c); c=c[g[c]!=g[i]]
        if len(c)==0: continue
        d=np.linalg.norm(Zf[c]-Zf[i],axis=1); j=np.argmin(d); lab[i]=y[c[j]]; dist[i]=d[j]
    return lab,dist
res={}
for s in ['155','2026','1029']:
    f=el[f'primary_fold_{s}'].values; p=O[f'A_base_{s}']
    for k in range(5):
        vam=f==k; lab,dist=nnfeat(f!=k)
        a0=roc_auc_score(y[vam],p[vam])
        for t in [0.5,1.0,1.5]:
            for w in [0.3,0.5,0.8]:
                p2=p.copy(); m=vam&(dist<t); p2[m]=(1-w)*p[m]+w*lab[m]
                res.setdefault((t,w),[]).append(roc_auc_score(y[vam],p2[vam])-a0)
        res.setdefault(('n<0.5',0),[]).append((vam&(dist<0.5)).sum()); res.setdefault(('n<1',0),[]).append((vam&(dist<1)).sum()); res.setdefault(('n<1.5',0),[]).append((vam&(dist<1.5)).sum())
for k,v in res.items():
    v=np.array(v)
    if isinstance(k[0],str): print(f'  {k[0]}: mean rows per validation fold {v.mean():.1f}')
    else: print(f'  t={k[0]} w={k[1]}: mean AUC gain {v.mean():+.4f} +- {v.std(ddof=1)/np.sqrt(15):.4f}')
# how do near-duplicate pairs look: same key/mode/time_signature? duration ratio?
print('\n=== what are the d<1 same-date different-group pairs?')
P=np.load('same_date_pairs.npy'); D=np.load('same_date_dist.npy'); m=D<1
a,b=P[m,0],P[m,1]
print('  n pairs',m.sum(),' same key',(el.key.values[a]==el.key.values[b]).mean().round(2),' same mode',(el['mode'].values[a]==el['mode'].values[b]).mean().round(2),' |dur ratio-1|<0.02',(np.abs(el.duration_ms.values[a]/el.duration_ms.values[b]-1)<0.02).mean().round(2),' |tempo diff|<1',(np.abs(el.tempo.values[a]-el.tempo.values[b])<1).mean().round(2),' agree',(y[a]==y[b]).mean().round(3))
m2=(D>=1)&(D<2)
a,b=P[m2,0],P[m2,1]
print('  1<=d<2: n',m2.sum(),' same key',(el.key.values[a]==el.key.values[b]).mean().round(2),' |dur ratio-1|<0.02',(np.abs(el.duration_ms.values[a]/el.duration_ms.values[b]-1)<0.02).mean().round(2),' agree',(y[a]==y[b]).mean().round(3))
# test: identical audio, different uri
sig=te[AUDIO].round(6).astype(str).agg('|'.join,axis=1); vc=sig.value_counts(); rep=vc[vc>1]
print('\n=== test: identical audio signatures (any uri):',len(rep),'rows',rep.sum(),'; of which different-uri:',sum(te[sig==s].uri.nunique()>1 for s in rep.index))
# dur_round in test and by era
print('dur_round share: train',(el.duration_ms%1000==0).mean().round(4),' test',(te.duration_ms%1000==0).mean().round(4))
era=pd.cut(el.year,[0,1989,1999,2009,2019,2030],labels=['<1990','1990s','2000s','2010s','2020+'])
print('High rate dur_round vs not, by era:'); print(el.assign(r=(el.duration_ms%1000==0)).groupby([era,'r'])['y'].agg(['size','mean']).round(3).unstack().to_string())
# expected test High rate under the removal mechanism
print('\nexpected test High rate: singleton-like rows',round(y[nc==1].mean(),3),' corrected for ~103 hidden multi-copy rows:',round((y[nc==1].sum()-103*0.79)/((nc==1).sum()-103),3),' -> test approx',round(((863-22)*((y[nc==1].sum()-103*0.79)/((nc==1).sum()-103))+22*0.79)/863,3))
