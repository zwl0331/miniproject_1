import pandas as pd, numpy as np, time, json, os, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
P='/Users/zwl0331/Desktop/Mini_project_1/'
AUD=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
D='track_album_release_date'
SEEDS=['155','2026','1029']

def hk(df,cols): return pd.util.hash_pandas_object(df[cols], index=False).values

def load():
    tr=pd.read_csv(P+'data/train.csv'); te=pd.read_csv(P+'data/test.csv'); sp=pd.read_csv(P+'artifacts/splits/production_v1.csv')
    tr['y']=(tr.Popularity_Type=='High').astype(int)
    tr['grp']=sp.primary_group.values; tr['elig']=sp.eligible.values
    for s in SEEDS: tr['fold_'+s]=sp['primary_fold_'+s].values
    tr['is_test']=0; te['is_test']=1; te['y']=-1; te['grp']=-1
    al=pd.concat([tr,te],ignore_index=True)
    # label-free global features (train+test)
    al['key_exact']=hk(al,AUD+[D]); al['key_audio']=hk(al,AUD)
    al['cnt_exact']=al.key_exact.map(al.key_exact.value_counts())
    al['cnt_audio']=al.key_audio.map(al.key_audio.value_counts())
    al['date_cnt_all']=al[D].map(al[D].value_counts())
    s=al[D].astype(str)
    al['prec']=s.str.len().map({10:2,7:1,4:0})
    al['year']=s.str[:4].astype(int)
    al['month']=np.where(al.prec>=1, s.str[5:7].where(al.prec>=1,'1').astype(int), np.nan)
    al['day']=np.where(al.prec>=2, s.str[8:10].where(al.prec>=2,'1').astype(int), np.nan)
    full=pd.to_datetime(np.where(al.prec==2,s,np.where(al.prec==1,s+'-01',s+'-01-01')),errors='coerce')
    al['dse']=(full-pd.Timestamp('1970-01-01')).days
    al['doy']=np.where(al.prec==2, full.dayofyear, np.nan)
    al['wday']=np.where(al.prec==2, full.dayofweek, np.nan)
    al['dur_min']=al.duration_ms/60000.
    tr=al[al.is_test==0].copy(); te=al[al.is_test==1].copy()
    tr=tr[tr.elig==1].copy().reset_index(drop=True)
    return tr,te

BASE=AUD+['year','month','day','doy','wday','prec','dse']

def te_encode(tr_df, va_df, col, y, m=3.0, inner=5, seed=0, prior=None):
    """cross-fitted target encoding of `col`. Returns (enc_tr, enc_va, n_tr, n_va)."""
    pr = y.mean() if prior is None else prior
    enc_tr=np.full(len(tr_df),np.nan); n_tr=np.zeros(len(tr_df))
    gkf=GroupKFold(n_splits=inner)
    for a,b in gkf.split(tr_df, groups=tr_df.grp.values):
        st=pd.DataFrame({'c':tr_df[col].values[a],'y':y[a]}).groupby('c').y.agg(['sum','count'])
        cb=tr_df[col].values[b]
        ssum=pd.Series(cb).map(st['sum']).fillna(0).values; scnt=pd.Series(cb).map(st['count']).fillna(0).values
        enc_tr[b]=(ssum+m*pr)/(scnt+m); n_tr[b]=scnt
    st=pd.DataFrame({'c':tr_df[col].values,'y':y}).groupby('c').y.agg(['sum','count'])
    cv=va_df[col].values
    ssum=pd.Series(cv).map(st['sum']).fillna(0).values; scnt=pd.Series(cv).map(st['count']).fillna(0).values
    enc_va=(ssum+m*pr)/(scnt+m); n_va=scnt
    return enc_tr,enc_va,n_tr,n_va

def run_cv(tr, make_model, feats, fold_feats=None, name='', log='results.csv', seeds=SEEDS, fit_kw=None, verbose=True, return_oof=False):
    """fold_feats: function(tr_df, va_df, y_tr) -> (Xtr_extra DataFrame, Xva_extra DataFrame)"""
    t0=time.time(); aucs=[]; aucs_single=[]; per_seed={}
    oof={s:np.zeros(len(tr)) for s in seeds}
    for s in seeds:
        fa=[]; fs=[]
        for f in range(5):
            trm=tr['fold_'+s].values!=f; vam=~trm
            trd=tr[trm]; vad=tr[vam]; ytr=trd.y.values; yva=vad.y.values
            Xtr=trd[feats].copy(); Xva=vad[feats].copy()
            if fold_feats is not None:
                etr,eva=fold_feats(trd,vad,ytr)
                Xtr=pd.concat([Xtr.reset_index(drop=True),etr.reset_index(drop=True)],axis=1)
                Xva=pd.concat([Xva.reset_index(drop=True),eva.reset_index(drop=True)],axis=1)
            mdl=make_model()
            if fit_kw: mdl.fit(Xtr,ytr,**fit_kw)
            else: mdl.fit(Xtr,ytr)
            p=mdl.predict_proba(Xva)[:,1]
            oof[s][vam]=p
            fa.append(roc_auc_score(yva,p))
            sm=vad.cnt_exact.values==1
            fs.append(roc_auc_score(yva[sm],p[sm]))
        per_seed[s]=np.mean(fa); aucs+=fa; aucs_single+=fs
    res={'name':name,'mean15':np.mean(aucs),'s155':per_seed['155'],'s2026':per_seed['2026'],'s1029':per_seed['1029'],'single15':np.mean(aucs_single),'std_fold':np.std(aucs),'secs':time.time()-t0,'nfeat':Xtr.shape[1]}
    if verbose: print(f"{name:45s} mean15={res['mean15']:.4f} seeds=({res['s155']:.4f},{res['s2026']:.4f},{res['s1029']:.4f}) single={res['single15']:.4f} sd={res['std_fold']:.3f} nfeat={res['nfeat']} {res['secs']:.0f}s", flush=True)
    hdr=not os.path.exists(log)
    pd.DataFrame([res]).to_csv(log,mode='a',header=hdr,index=False)
    if return_oof: return res,oof
    return res
