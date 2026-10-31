import sys, pickle; sys.path.insert(0,'.')
from pipe import *
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
tr,te=load()
F=BASE+['cnt_exact','date_cnt_all']; FN=BASE+['date_cnt_all']
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
def ff(trd,vad,ytr):
    x,y_,na,nb=te_encode(trd,vad,D,ytr,m=3.0); k1,k2=knn_feat(trd,vad,ytr,20,AUD+['year'])
    return pd.DataFrame({'te_date':x,'n_date':na,'knn20':k1}),pd.DataFrame({'te_date':y_,'n_date':nb,'knn20':k2})
class SK:
    def __init__(self,est): self.p=make_pipeline(SimpleImputer(strategy='median'),est)
    def fit(self,X,y): self.p.fit(X,y); return self
    def predict_proba(self,X): return self.p.predict_proba(X)
OOF=pickle.load(open('oof5.pkl','rb'))
r,o=run_cv(tr,lgbm(),FN,fold_feats=ff,name='lgb NO cnt (oof)',return_oof=True); OOF['lgb_nocnt']=o
r,o=run_cv(tr,lambda: SK(ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=0.5,n_jobs=2,random_state=0)),FN,fold_feats=ff,name='et NO cnt (oof)',return_oof=True); OOF['et_nocnt']=o
pickle.dump(OOF,open('oof8.pkl','wb'))
y=tr.y.values; cnt=tr.cnt_exact.values; sm=cnt==1
rng=np.random.default_rng(0)
def testlike(pred_fn, draws=20):
    """AUC on singletons + dup rows subsampled to 2.5% of the subset (test composition); mean over folds/seeds/draws"""
    out=[]; outs=[]
    for s in SEEDS:
        for f in range(5):
            m=tr['fold_'+s].values==f; p=pred_fn(s,m); ym=y[m]; smm=sm[m]; cm=cnt[m]
            outs.append(roc_auc_score(ym[smm],p[smm]))
            dup_idx=np.where(~smm)[0]; n_dup=int(round(0.0255*smm.sum()/(1-0.0255)))
            for _ in range(draws):
                pick=rng.choice(dup_idx,n_dup,replace=False); keep=np.concatenate([np.where(smm)[0],pick])
                out.append(roc_auc_score(ym[keep],p[keep]))
    return np.mean(out),np.mean(outs)
def P(name): return lambda s,m: OOF[name][s][m]
def blend(ws): return lambda s,m: sum(w*rankdata(OOF[n][s][m]) for n,w in ws.items())
def boost(ws, add=0.5):
    # no-cnt model, then push rows with cnt>1 up: rank-normalize, add constant for dup rows
    def fn(s,m):
        p=sum(w*rankdata(OOF[n][s][m]) for n,w in ws.items()); p=p/p.max(); return p+add*(cnt[m]>1)
    return fn
for nm,fn in [('lgb cnt',P('lgb')),('lgb no-cnt',P('lgb_nocnt')),('lgb+0.5et cnt',blend({'lgb':1,'et':0.5})),('lgb+0.5et no-cnt',blend({'lgb_nocnt':1,'et_nocnt':0.5})),
              ('lgb+0.5et no-cnt + dup boost',boost({'lgb_nocnt':1,'et_nocnt':0.5})),('lgb no-cnt + dup boost',boost({'lgb_nocnt':1}))]:
    a,b=testlike(fn); print(f'{nm:35s} test-like AUC={a:.4f}  singleton AUC={b:.4f}')
