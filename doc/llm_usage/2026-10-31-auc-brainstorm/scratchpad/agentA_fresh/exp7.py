import sys, pickle; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
tr,te=load()
F=BASE+['cnt_exact','date_cnt_all']
def lgbm(**kw):
    p=dict(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,num_threads=2,verbose=-1,random_state=0)
    p.update(kw); return lambda: lgb.LGBMClassifier(**p)
def knn_feat(trd,vad,ytr,k,cols):
    sc=StandardScaler().fit(trd[cols].values); A=sc.transform(trd[cols].values); B=sc.transform(vad[cols].values)
    out_tr=np.zeros(len(trd)); gkf=GroupKFold(5)
    for a,b in gkf.split(A,groups=trd.grp.values):
        nn=NearestNeighbors(n_neighbors=k).fit(A[a]); _,idx=nn.kneighbors(A[b]); out_tr[b]=ytr[a][idx].mean(1)
    nn=NearestNeighbors(n_neighbors=k).fit(A); _,idx=nn.kneighbors(B); out_va=ytr[idx].mean(1)
    return out_tr,out_va
def mk(m,k):
    def ff(trd,vad,ytr):
        x,y_,na,nb=te_encode(trd,vad,D,ytr,m=m); k1,k2=knn_feat(trd,vad,ytr,k,AUD+['year'])
        return pd.DataFrame({'te_date':x,'n_date':na,'knn':k1}),pd.DataFrame({'te_date':y_,'n_date':nb,'knn':k2})
    return ff
OOF={}
run_cv(tr,lgbm(),F,fold_feats=mk(1.0,20),name='lgb teDate m=1 knn20')
run_cv(tr,lgbm(),F,fold_feats=mk(10.0,20),name='lgb teDate m=10 knn20')
run_cv(tr,lgbm(),F,fold_feats=mk(3.0,10),name='lgb teDate m=3 knn10')
r,o=run_cv(tr,lgbm(),F,fold_feats=mk(3.0,35),name='lgb teDate m=3 knn35',return_oof=True); OOF['lgb_k35']=o
# drop cnt_exact entirely (singleton-focused check) 
run_cv(tr,lgbm(),BASE+['date_cnt_all'],fold_feats=mk(3.0,20),name='lgb teDate knn20 NO cnt')
pickle.dump(OOF,open('oof7.pkl','wb'))
