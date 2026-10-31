import numpy as np
from scipy.stats import rankdata
rng=np.random.default_rng(3)
def hm_se(A,n,p=0.35):
    n1=round(n*p); n0=n-n1; Q1=A/(2-A); Q2=2*A*A/(1+A)
    return np.sqrt((A*(1-A)+(n1-1)*(Q1-A*A)+(n0-1)*(Q2-A*A))/(n1*n0))
def auc(y,s):
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print('Hanley-McNeil SE, AUC 0.84, positive rate 0.35:')
for frac,n in [(0.2,173),(0.3,259),(0.4,345),(0.5,432),(0.6,518),(1.0,863)]: print(f'  {frac:.0%} n={n}: SE={hm_se(0.84,n):.4f}  95% half-width {1.96*hm_se(0.84,n):.3f}')
Z=np.load('oof_probe.npz'); y=Z['y']
pA=np.mean([Z[f'A_base_{s}'] for s in ['155','2026','1029']],axis=0); pE=np.mean([Z[f'E_wmean_{s}'] for s in ['155','2026','1029']],axis=0); pB=np.mean([Z[f'B_te_{s}'] for s in ['155','2026','1029']],axis=0)
print(f'OOF seed-averaged AUC: A={auc(y,pA):.4f} B={auc(y,pB):.4f} E={auc(y,pE):.4f}; corr(A,E)={np.corrcoef(pA,pE)[0,1]:.3f} corr(A,B)={np.corrcoef(pA,pB)[0,1]:.3f}')
for n in [345,518,863]:
    a=[];dd=[]
    for _ in range(4000):
        ii=rng.choice(len(y),n,replace=False); a.append(auc(y[ii],pA[ii])); dd.append(auc(y[ii],pE[ii])-auc(y[ii],pA[ii]))
    print(f'  empirical subsample of OOF, n={n}: SE(AUC_A)={np.std(a):.4f}; SE(AUC_E - AUC_A)={np.std(dd):.4f}  [pred corr {np.corrcoef(pA,pE)[0,1]:.3f}]')
