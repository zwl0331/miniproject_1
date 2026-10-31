import sys; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb
tr,te=load()
print(len(tr), tr.y.mean())
def lgbm(**kw):
    p=dict(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,num_threads=2,verbose=-1,random_state=0)
    p.update(kw); return lambda: lgb.LGBMClassifier(**p)
r0=run_cv(tr,lgbm(),BASE,name='lgb_base')
r1=run_cv(tr,lgbm(),BASE+['cnt_exact'],name='lgb_base+cnt')
r2=run_cv(tr,lgbm(),BASE+['cnt_exact','date_cnt_all'],name='lgb_base+cnt+datecnt')
def ff_date(trd,vad,ytr):
    a,b,na,nb=te_encode(trd,vad,D,ytr,m=3.0)
    return pd.DataFrame({'te_date':a,'n_date_tr':na}), pd.DataFrame({'te_date':b,'n_date_tr':nb})
r3=run_cv(tr,lgbm(),BASE+['cnt_exact','date_cnt_all'],fold_feats=ff_date,name='lgb_base+cnt+datecnt+teDate')
