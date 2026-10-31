import sys; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb
tr,te=load()
# row-order artifact check: autocorrelation of label along the original row order; test ID order
y=tr.y.values
print('lag-1 label autocorr along train row order:', np.corrcoef(y[:-1],y[1:])[0,1])
blk=pd.Series(y).groupby(np.arange(len(y))//200).mean(); print('block(200) High rate along train order:', blk.round(2).tolist())
print('test ID vs year corr:', np.corrcoef(te.ID.values, te.year.values)[0,1])
def lgbm(**kw):
    p=dict(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,num_threads=2,verbose=-1,random_state=0)
    p.update(kw); return lambda: lgb.LGBMClassifier(**p)
def ff_date(trd,vad,ytr):
    a,b,na,nb=te_encode(trd,vad,D,ytr,m=3.0)
    return pd.DataFrame({'te_date':a,'n_date_tr':na}), pd.DataFrame({'te_date':b,'n_date_tr':nb})
F=BASE+['cnt_exact','date_cnt_all']
# Variant: weight duplicates so each track counts once (sample_weight=1/cnt within train)
class W:
    def __init__(self,mk): self.m=mk()
    def fit(self,X,y):
        w=1.0/X['cnt_exact'].values; self.m.fit(X,y,sample_weight=w); return self
    def predict_proba(self,X): return self.m.predict_proba(X)
run_cv(tr,lambda: W(lgbm()),F,fold_feats=ff_date,name='lgb+teDate w=1/cnt')
# Variant: more regularized / smaller trees / more trees
run_cv(tr,lgbm(num_leaves=7,n_estimators=600),F,fold_feats=ff_date,name='lgb+teDate nl7 n600')
run_cv(tr,lgbm(num_leaves=31,min_child_samples=40,n_estimators=300),F,fold_feats=ff_date,name='lgb+teDate nl31 mcs40 n300')
run_cv(tr,lgbm(learning_rate=0.015,n_estimators=800),F,fold_feats=ff_date,name='lgb+teDate lr.015 n800')
run_cv(tr,lgbm(extra_trees=True,n_estimators=800),F,fold_feats=ff_date,name='lgb+teDate extra_trees n800')
run_cv(tr,lgbm(colsample_bytree=0.5),F,fold_feats=ff_date,name='lgb+teDate colsample.5')
run_cv(tr,lgbm(min_child_samples=50,reg_lambda=5),F,fold_feats=ff_date,name='lgb+teDate mcs50 l2=5')
