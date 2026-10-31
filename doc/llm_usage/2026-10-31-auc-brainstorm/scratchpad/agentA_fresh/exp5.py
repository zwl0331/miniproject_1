import sys, pickle; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
tr,te=load()
tr['ym']=tr[D].astype(str).str[:7]; te['ym']=te[D].astype(str).str[:7]
def lgbm(**kw):
    p=dict(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,num_threads=2,verbose=-1,random_state=0)
    p.update(kw); return lambda: lgb.LGBMClassifier(**p)
F=BASE+['cnt_exact','date_cnt_all']
def knn_feat(trd,vad,ytr,k,cols,wyear=1.0):
    sc=StandardScaler().fit(trd[cols].values); A=sc.transform(trd[cols].values); B=sc.transform(vad[cols].values)
    if 'year' in cols: j=cols.index('year'); A[:,j]*=wyear; B[:,j]*=wyear
    out_tr=np.zeros(len(trd)); gkf=GroupKFold(5)
    for a,b in gkf.split(A,groups=trd.grp.values):
        nn=NearestNeighbors(n_neighbors=k).fit(A[a]); _,idx=nn.kneighbors(A[b]); out_tr[b]=ytr[a][idx].mean(1)
    nn=NearestNeighbors(n_neighbors=k).fit(A); _,idx=nn.kneighbors(B); out_va=ytr[idx].mean(1)
    return out_tr,out_va
def mk_ff(knn_specs, te_cols=(D,), ms=(3.0,)):
    def ff(trd,vad,ytr):
        a={}; b={}
        for c,m in zip(te_cols,ms):
            x,y_,na,nb=te_encode(trd,vad,c,ytr,m=m); a['te_'+c]=x; b['te_'+c]=y_; a['n_'+c]=na; b['n_'+c]=nb
        for nm,(k,cols,w) in knn_specs.items():
            x,y_=knn_feat(trd,vad,ytr,k,cols,w); a[nm]=x; b[nm]=y_
        return pd.DataFrame(a),pd.DataFrame(b)
    return ff
K20={'knn20':(20,AUD+['year'],1.0)}
OOF={}
r,o=run_cv(tr,lgbm(),F,fold_feats=mk_ff(K20,(D,'ym'),(3.0,5.0)),name='lgb+teDate+teYM+knn20',return_oof=True)
r,o=run_cv(tr,lgbm(),F,fold_feats=mk_ff({'knn20':(20,AUD+['year'],1.0),'knn50':(50,AUD+['year'],1.0),'knn20y3':(20,AUD+['year'],3.0),'knn20a':(20,AUD,1.0)}),name='lgb+teDate+knn x4',return_oof=True)
r,o=run_cv(tr,lgbm(),F,fold_feats=mk_ff(K20),name='lgb+teDate+knn20',return_oof=True); OOF['lgb']=o
# diverse models on imputed/standardized features
NUM=F+['te_'+D,'n_'+D,'knn20']
class SK:
    def __init__(self,est): self.p=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),est)
    def fit(self,X,y): self.p.fit(X,y); return self
    def predict_proba(self,X): return self.p.predict_proba(X)
r,o=run_cv(tr,lambda: SK(ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=0.5,n_jobs=2,random_state=0)),F,fold_feats=mk_ff(K20),name='extratrees+teDate+knn20',return_oof=True); OOF['et']=o
r,o=run_cv(tr,lambda: SK(make_pipeline(SplineTransformer(n_knots=6,degree=3),LogisticRegression(C=0.3,max_iter=2000))),F,fold_feats=mk_ff(K20),name='spline-logreg+teDate+knn20',return_oof=True); OOF['lr']=o
r,o=run_cv(tr,lambda: SK(KNeighborsClassifier(n_neighbors=40,weights='distance',n_jobs=2)),F,fold_feats=mk_ff(K20),name='knn40clf+teDate',return_oof=True); OOF['knn']=o
pickle.dump(OOF,open('oof5.pkl','wb'))
