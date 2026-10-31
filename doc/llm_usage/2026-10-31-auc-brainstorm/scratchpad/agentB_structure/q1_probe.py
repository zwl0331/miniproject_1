from common import *
import lightgbm as lgb, time, json
from sklearn.metrics import roc_auc_score
tr,te,sp=load()
tr=tr.join(sp[['eligible','primary_group','primary_fold_155','primary_fold_2026','primary_fold_1029']])
el=tr[tr.eligible==1].reset_index(drop=True)
Z=zfeat(el).values
y=el.y.values; g=el.primary_group.values; ds=el.date_str.values
dparts=pd.DataFrame({'year':el.year,'month':el.month,'day':el.day,'date_len':el.date_len,'dow':el.dow,
                     'ordinal':pd.to_datetime(el.date_str.where(el.date_len==10, el.date_str.str[:4]+'-07-01'),errors='coerce').map(lambda t: t.toordinal() if pd.notna(t) else np.nan)})
Xbase=pd.concat([el[AUDIO].reset_index(drop=True),dparts.reset_index(drop=True)],axis=1)
PARAMS=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=20,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=2,seed=1)
NROUND=500
def neigh_feats(train_mask, prior, h=1.5, m_te=3.0):
    """for every row, features from training-part rows with the same date string and a different group"""
    n=len(el); out=dict(nn_label=np.full(n,np.nan),nn_dist=np.full(n,np.nan),n_mates=np.zeros(n),mean_mate=np.full(n,np.nan),
                        te_m3=np.full(n,prior),wmean=np.full(n,np.nan),k3mean=np.full(n,np.nan),nn2_label=np.full(n,np.nan),min_dist_H=np.full(n,np.nan),min_dist_L=np.full(n,np.nan))
    tr_idx=np.where(train_mask)[0]
    bydate={}
    for i in tr_idx: bydate.setdefault(ds[i],[]).append(i)
    for i in range(n):
        cand=bydate.get(ds[i])
        if not cand: continue
        cand=np.array(cand); cand=cand[g[cand]!=g[i]]
        if len(cand)==0: continue
        d=np.linalg.norm(Z[cand]-Z[i],axis=1); o=np.argsort(d); d=d[o]; yc=y[cand[o]]
        out['nn_label'][i]=yc[0]; out['nn_dist'][i]=d[0]; out['n_mates'][i]=len(cand); out['mean_mate'][i]=yc.mean()
        out['te_m3'][i]=(yc.sum()+m_te*prior)/(len(cand)+m_te)
        w=np.exp(-(d**2)/(2*h*h)); out['wmean'][i]=(w*yc).sum()/w.sum() if w.sum()>0 else yc.mean()
        out['k3mean'][i]=yc[:3].mean()
        if len(cand)>1: out['nn2_label'][i]=yc[1]
        if (yc==1).any(): out['min_dist_H'][i]=d[yc==1].min()
        if (yc==0).any(): out['min_dist_L'][i]=d[yc==0].min()
    return pd.DataFrame(out)
SETS={'A_base':[], 'B_te':['te_m3','n_mates'], 'C_nn':['nn_label','nn_dist','n_mates'], 'D_all':['te_m3','n_mates','nn_label','nn_dist','wmean','k3mean','min_dist_H','min_dist_L'],
      'E_wmean':['wmean','n_mates','nn_dist']}
res={k:[] for k in SETS}; single={'nn_label':[],'wmean':[],'te_m3':[],'mean_mate':[],'k3mean':[]}
oof={k:{s:np.zeros(len(el)) for s in ['155','2026','1029']} for k in SETS}
cover=[]
t0=time.time()
for s in ['155','2026','1029']:
    f=el[f'primary_fold_{s}'].values
    for k in range(5):
        trm=f!=k; vam=f==k
        prior=y[trm].mean()
        F=neigh_feats(trm,prior)
        cover.append(dict(seed=s,fold=k,val_with_mate=np.isfinite(F.nn_label[vam]).mean(),train_with_mate=np.isfinite(F.nn_label[trm]).mean(),
                          val_near1=(F.nn_dist[vam]<1.0).mean(),val_near2=(F.nn_dist[vam]<2.34).mean(),train_near2=(F.nn_dist[trm]<2.34).mean()))
        for name in single:
            v=F[name][vam].fillna(prior).values; single[name].append(roc_auc_score(y[vam],v))
        for name,cols in SETS.items():
            X=pd.concat([Xbase,F[cols]],axis=1) if cols else Xbase
            mdl=lgb.train(PARAMS,lgb.Dataset(X[trm],y[trm]),NROUND)
            p=mdl.predict(X[vam]); oof[name][s][vam]=p
            res[name].append(roc_auc_score(y[vam],p))
        print(s,k,{n:round(res[n][-1],4) for n in SETS},f'{time.time()-t0:.0f}s',flush=True)
print('\n=== single-feature AUC on validation rows (NaN -> training prior), mean of 15 folds')
for n,v in single.items(): print(f'  {n:10s} {np.mean(v):.4f} +- {np.std(v,ddof=1)/np.sqrt(15):.4f}')
print('\n=== LightGBM probe, mean of 15 fold AUCs (SE over folds of the paired difference vs A)')
a=np.array(res['A_base'])
for n in SETS:
    v=np.array(res[n]); d=v-a
    print(f'  {n:8s} {v.mean():.4f}  diff {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(15):.4f}   per-seed means {[round(v[i*5:(i+1)*5].mean(),4) for i in range(3)]}')
C=pd.DataFrame(cover); print('\ncoverage (fraction of rows with a same-date mate in the training part):'); print(C.mean(numeric_only=True).round(3).to_string())
np.savez('oof_probe.npz',y=y,**{f'{k}_{s}':oof[k][s] for k in oof for s in oof[k]})
json.dump({k:[float(x) for x in v] for k,v in res.items()},open('q1_probe_res.json','w'))
