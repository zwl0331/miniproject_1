import sys, pickle, itertools; sys.path.insert(0,'.')
from pipe import *
from scipy.stats import rankdata
tr,te=load()
OOF=pickle.load(open('oof5.pkl','rb'))
for f in sys.argv[1:]:
    OOF.update(pickle.load(open(f,'rb')))
names=list(OOF); print('models:',names)
y=tr.y.values; sm=tr.cnt_exact.values==1
def score(w):
    aucs=[]; sing=[]; per={}
    for s in SEEDS:
        fa=[]
        for f in range(5):
            m=tr['fold_'+s].values==f
            p=sum(w[n]*rankdata(OOF[n][s][m]) for n in names if w.get(n,0)>0)
            fa.append(roc_auc_score(y[m],p)); sing.append(roc_auc_score(y[m&sm],p[sm[m]]))
        per[s]=np.mean(fa); aucs+=fa
    return np.mean(aucs),per,np.mean(sing)
rows=[]
for w in [{'lgb':1},{'lgb':1,'et':0.5},{'lgb':1,'et':1},{'lgb':1,'lr':0.3},{'lgb':1,'knn':0.3},{'lgb':1,'et':0.5,'lr':0.2},{'lgb':1,'et':0.5,'lr':0.2,'knn':0.2},
          {'lgb':1,'xgb':1},{'lgb':1,'xgb':1,'et':0.5},{'lgb':1,'ebm':1},{'lgb':1,'xgb':1,'ebm':1},{'lgb':1,'xgb':1,'ebm':1,'et':0.5},{'lgb':1,'xgb':1,'ebm':1,'et':0.5,'lr':0.2}]:
    if any(n not in OOF for n in w): continue
    m,per,sg=score(w); rows.append((str(w),m,per['155'],per['2026'],per['1029'],sg))
    print(f"{str(w):55s} mean15={m:.4f} seeds=({per['155']:.4f},{per['2026']:.4f},{per['1029']:.4f}) single={sg:.4f}")
pd.DataFrame(rows,columns=['blend','mean15','s155','s2026','s1029','single15']).to_csv('blend_results.csv',index=False)
