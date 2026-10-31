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
def ff_date(trd,vad,ytr):
    a,b,na,nb=te_encode(trd,vad,D,ytr,m=3.0)
    return pd.DataFrame({'te_date':a,'n_date_tr':na}), pd.DataFrame({'te_date':b,'n_date_tr':nb})
F=BASE+['cnt_exact','date_cnt_all']
# engineered audio features
for df in (tr,te):
    df['log_dur']=np.log(df.duration_ms); df['log_instr']=np.log10(df.instrumentalness+1e-6); df['log_speech']=np.log(df.speechiness)
    df['log_live']=np.log(df.liveness); df['loud_energy']=df.loudness-20*np.log10(df.energy+1e-3)
    df['dance_val']=df.danceability*df.valence; df['acou_energy']=df.acousticness*df.energy
    df['instr_zero']=(df.instrumentalness<1e-4).astype(int); df['age']=2024.9-(df.dse/365.25+1970)
yr_med=tr.groupby('year').loudness.median()
for df in (tr,te): df['loud_rel_year']=df.loudness-df.year.map(yr_med).fillna(tr.loudness.median())
ENG=['log_instr','loud_energy','dance_val','acou_energy','instr_zero','loud_rel_year']
run_cv(tr,lgbm(),F+ENG,fold_feats=ff_date,name='lgb+teDate+ENG')
# kNN target encoding in standardized audio space (cross-fitted), k=20, plus same-year-window variant
def ff_knn(trd,vad,ytr,k=20):
    cols=AUD+['year']
    sc=StandardScaler().fit(trd[cols].values); A=sc.transform(trd[cols].values); B=sc.transform(vad[cols].values)
    out_tr=np.zeros(len(trd)); gkf=GroupKFold(5)
    for a,b in gkf.split(A,groups=trd.grp.values):
        nn=NearestNeighbors(n_neighbors=k).fit(A[a]); _,idx=nn.kneighbors(A[b]); out_tr[b]=ytr[a][idx].mean(1)
    nn=NearestNeighbors(n_neighbors=k).fit(A); _,idx=nn.kneighbors(B); out_va=ytr[idx].mean(1)
    a,b,na,nb=te_encode(trd,vad,D,ytr,m=3.0)
    return pd.DataFrame({'te_date':a,'n_date_tr':na,'knn20':out_tr}), pd.DataFrame({'te_date':b,'n_date_tr':nb,'knn20':out_va})
run_cv(tr,lgbm(),F,fold_feats=ff_knn,name='lgb+teDate+knn20')
# CatBoost with date string as categorical (native ordered TS) + te features
class CB:
    def __init__(self,**kw):
        self.p=dict(iterations=800,learning_rate=0.03,depth=6,l2_leaf_reg=3,thread_count=2,verbose=0,random_seed=0); self.p.update(kw)
    def fit(self,X,y):
        self.cat=[c for c in X.columns if X[c].dtype==object]
        self.m=CatBoostClassifier(**self.p,cat_features=self.cat); self.m.fit(X,y); return self
    def predict_proba(self,X): return self.m.predict_proba(X)
def ff_cat(trd,vad,ytr):
    return pd.DataFrame({'date_cat':trd[D].astype(str).values}), pd.DataFrame({'date_cat':vad[D].astype(str).values})
run_cv(tr,lambda: CB(),F,fold_feats=ff_date,name='cat d6 +teDate')
run_cv(tr,lambda: CB(),F,fold_feats=ff_cat,name='cat d6 +dateCat')
