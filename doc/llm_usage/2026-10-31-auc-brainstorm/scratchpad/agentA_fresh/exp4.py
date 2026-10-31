import sys; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
tr,te=load()
def lgbm(**kw):
    p=dict(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,num_threads=2,verbose=-1,random_state=0)
    p.update(kw); return lambda: lgb.LGBMClassifier(**p)
F=BASE+['cnt_exact','date_cnt_all']
def ff_knn(trd,vad,ytr,k=20):
    cols=AUD+['year']
    sc=StandardScaler().fit(trd[cols].values); A=sc.transform(trd[cols].values); B=sc.transform(vad[cols].values)
    out_tr=np.zeros(len(trd)); gkf=GroupKFold(5)
    for a,b in gkf.split(A,groups=trd.grp.values):
        nn=NearestNeighbors(n_neighbors=k).fit(A[a]); _,idx=nn.kneighbors(A[b]); out_tr[b]=ytr[a][idx].mean(1)
    nn=NearestNeighbors(n_neighbors=k).fit(A); _,idx=nn.kneighbors(B); out_va=ytr[idx].mean(1)
    a,b,na,nb=te_encode(trd,vad,D,ytr,m=3.0)
    return pd.DataFrame({'te_date':a,'n_date_tr':na,'knn20':out_tr}), pd.DataFrame({'te_date':b,'n_date_tr':nb,'knn20':out_va})
# error analysis with OOF of lgb+teDate+knn20
res,oof=run_cv(tr,lgbm(),F,fold_feats=ff_knn,name='lgb+teDate+knn20 (oof)',return_oof=True)
p=np.mean([oof[s] for s in SEEDS],axis=0)
tr['oof']=p
np.save('oof_lgb_knn.npy',p)
tr['era']=pd.cut(tr.year,[1900,1999,2009,2015,2019,2021,2023,2025])
sm=tr.cnt_exact==1
print('singleton AUC by era (OOF, seed-avg):')
for e,g in tr[sm].groupby('era',observed=True):
    if g.y.nunique()>1: print(f'  {e}: n={len(g)} rate={g.y.mean():.2f} auc={roc_auc_score(g.y,g.oof):.3f}')
print('all singletons auc (seed-avg oof):', roc_auc_score(tr.y[sm],tr.oof[sm]), ' all rows:', roc_auc_score(tr.y,tr.oof))
# seed-bagged LGB (5 seeds)
class Bag:
    def __init__(self,mk,n=5): self.ms=[mk(random_state=i)() for i in range(n)]
    def fit(self,X,y):
        for m in self.ms: m.fit(X,y)
        return self
    def predict_proba(self,X): return np.mean([m.predict_proba(X) for m in self.ms],axis=0)
run_cv(tr,lambda: Bag(lgbm,5),F,fold_feats=ff_knn,name='lgb+teDate+knn20 bag5')
# CatBoost with date categorical (fixed dtype detection), depth 4
class CB:
    def __init__(self,**kw):
        self.p=dict(iterations=800,learning_rate=0.03,depth=4,l2_leaf_reg=3,thread_count=2,verbose=0,random_seed=0); self.p.update(kw)
    def fit(self,X,y):
        self.cat=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        X=X.copy()
        for c in self.cat: X[c]=X[c].astype(str)
        self.m=CatBoostClassifier(**self.p,cat_features=self.cat); self.m.fit(X,y); return self
    def predict_proba(self,X):
        X=X.copy()
        for c in self.cat: X[c]=X[c].astype(str)
        return self.m.predict_proba(X)
def ff_cat(trd,vad,ytr):
    a,b=ff_knn(trd,vad,ytr)
    a['date_cat']=trd[D].astype(str).values; b['date_cat']=vad[D].astype(str).values
    return a,b
run_cv(tr,lambda: CB(),F,fold_feats=ff_cat,name='cat d4 +teDate+knn20+dateCat')
run_cv(tr,lambda: CB(),F,fold_feats=ff_knn,name='cat d4 +teDate+knn20')
