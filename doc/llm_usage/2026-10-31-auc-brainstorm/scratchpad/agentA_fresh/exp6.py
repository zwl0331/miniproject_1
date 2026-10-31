import sys, pickle; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb, xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier
tr,te=load()
F=BASE+['cnt_exact','date_cnt_all']
def knn_feat(trd,vad,ytr,k,cols):
    sc=StandardScaler().fit(trd[cols].values); A=sc.transform(trd[cols].values); B=sc.transform(vad[cols].values)
    out_tr=np.zeros(len(trd)); gkf=GroupKFold(5)
    for a,b in gkf.split(A,groups=trd.grp.values):
        nn=NearestNeighbors(n_neighbors=k).fit(A[a]); _,idx=nn.kneighbors(A[b]); out_tr[b]=ytr[a][idx].mean(1)
    nn=NearestNeighbors(n_neighbors=k).fit(A); _,idx=nn.kneighbors(B); out_va=ytr[idx].mean(1)
    return out_tr,out_va
def ff(trd,vad,ytr):
    x,y_,na,nb=te_encode(trd,vad,D,ytr,m=3.0); k1,k2=knn_feat(trd,vad,ytr,20,AUD+['year'])
    return pd.DataFrame({'te_date':x,'n_date':na,'knn20':k1}),pd.DataFrame({'te_date':y_,'n_date':nb,'knn20':k2})
OOF={}
r,o=run_cv(tr,lambda: xgb.XGBClassifier(n_estimators=500,learning_rate=0.03,max_depth=4,min_child_weight=5,subsample=0.8,colsample_bytree=0.7,reg_lambda=2,n_jobs=2,random_state=0,tree_method='hist'),F,fold_feats=ff,name='xgb d4 +teDate+knn20',return_oof=True); OOF['xgb']=o
pickle.dump(OOF,open('oof6.pkl','wb'))
r,o=run_cv(tr,lambda: ExplainableBoostingClassifier(interactions=10,n_jobs=2,random_state=0,outer_bags=4,max_rounds=3000),F,fold_feats=ff,name='ebm +teDate+knn20',return_oof=True); OOF['ebm']=o
pickle.dump(OOF,open('oof6.pkl','wb'))
