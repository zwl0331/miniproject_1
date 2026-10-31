from common import *
import lightgbm as lgb, time, json
from sklearn.metrics import roc_auc_score
tr,te,sp=load()
tr=tr.join(sp[['eligible','primary_group','primary_fold_155','primary_fold_2026','primary_fold_1029']])
el=tr[tr.eligible==1].reset_index(drop=True); y=el.y.values
dparts=pd.DataFrame({'year':el.year,'month':el.month,'day':el.day,'date_len':el.date_len,'dow':el.dow,
                     'ordinal':pd.to_datetime(el.date_str.where(el.date_len==10, el.date_str.str[:4]+'-07-01'),errors='coerce').map(lambda t: t.toordinal() if pd.notna(t) else np.nan)})
dur_round=(el.duration_ms%1000==0).astype(int).rename('dur_round')
uvc=el.uri.value_counts(); ncopies=el.uri.map(uvc).rename('n_copies')
X=pd.concat([el[AUDIO].reset_index(drop=True),dparts,dur_round,ncopies],axis=1)
PARAMS=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=20,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=2,seed=1)
SETS={'audio_only':AUDIO,'audio_year':AUDIO+['year'],'audio_ordinal':AUDIO+['ordinal'],'audio_year_md':AUDIO+['year','month','day'],
      'audio_year_md_dow':AUDIO+['year','month','day','dow'],'full(A)':AUDIO+['year','month','day','date_len','dow','ordinal'],
      'full+dur_round':AUDIO+['year','month','day','date_len','dow','ordinal','dur_round'],
      'full+n_copies(LEAKY-ish)':AUDIO+['year','month','day','date_len','dow','ordinal','n_copies']}
res={k:[] for k in SETS}; res_single={k:[] for k in SETS}; t0=time.time()
for s in ['155','2026','1029']:
    f=el[f'primary_fold_{s}'].values
    for k in range(5):
        trm=f!=k; vam=f==k
        for name,cols in SETS.items():
            mdl=lgb.train(PARAMS,lgb.Dataset(X.loc[trm,cols],y[trm]),500); p=mdl.predict(X.loc[vam,cols])
            res[name].append(roc_auc_score(y[vam],p))
            # AUC on singleton-uri validation rows only (test-like population)
            sm=(ncopies.values[vam]==1); res_single[name].append(roc_auc_score(y[vam][sm],p[sm]))
        print(s,k,f'{time.time()-t0:.0f}s',flush=True)
a=np.array(res['full(A)'])
print('\n=== date-part ablation, mean of 15 fold AUCs; diff vs full(A) with SE over folds; last column = AUC on singleton-uri validation rows only')
for n in SETS:
    v=np.array(res[n]); d=v-a; print(f'  {n:26s} {v.mean():.4f}  diff {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(15):.4f}   singleton-only AUC {np.mean(res_single[n]):.4f}')
json.dump({'all':{k:[float(x) for x in v] for k,v in res.items()},'single':{k:[float(x) for x in v] for k,v in res_single.items()}},open('q_ablation.json','w'))
