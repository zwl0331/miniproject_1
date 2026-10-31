import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
rng=np.random.default_rng(42)
def auc(y,s):
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
p=0.35; d=np.sqrt(2)*stats.norm.ppf(0.84); vy=d*d*p*(1-p)
def noise_corr(rho): return rho*(vy+1)-vy
print('=== (b) paired difference, equal true AUC 0.84, by n and prediction correlation (2500 reps)')
rows=[]
for n in [345,518,863]:
    for rho in [0.8,0.9,0.97]:
        r=noise_corr(rho); out=[]
        for _ in range(2500):
            yy=(rng.random(n)<p).astype(int); e1=rng.standard_normal(n); e2=r*e1+np.sqrt(1-r*r)*rng.standard_normal(n)
            out.append((auc(yy,d*yy+e1),auc(yy,d*yy+e2)))
        out=np.array(out); rows.append(dict(n=n,pred_corr=rho,SE_single=round(out[:,0].std(),4),SE_diff=round((out[:,1]-out[:,0]).std(),4),AUC_corr=round(np.corrcoef(out[:,0],out[:,1])[0,1],3),gap_95=round(1.96*(out[:,1]-out[:,0]).std(),3)))
print(pd.DataFrame(rows).to_string(index=False))
print('\n=== (c) winner\'s curse on n=345: E[max of k public AUCs] - E[single], equal true AUC, pairwise prediction corr rho (2500 reps)')
rows=[]
for rho in [0.8,0.9]:
    r=noise_corr(rho)
    for k in [1,4,6,13,24]:
        out=[]
        for _ in range(2500):
            n=345; yy=(rng.random(n)<p).astype(int); z=rng.standard_normal(n); E=np.sqrt(r)*z[:,None]+np.sqrt(1-r)*rng.standard_normal((n,k)); S=d*yy[:,None]+E
            a=[auc(yy,S[:,j]) for j in range(k)]; out.append((max(a),a[0]))
        out=np.array(out); rows.append(dict(pred_corr=rho,k=k,E_max_minus_true=round(out[:,0].mean()-out[:,1].mean(),4),SD_max=round(out[:,0].std(),4)))
print(pd.DataFrame(rows).to_string(index=False))
print('\n=== (d) best-of-k2 (rival) minus best-of-k1 (team), equal TRUE AUC, n=345, within-team pred corr 0.9, between-team corr rb (2500 reps)')
def sim_gap(k1,k2,rb,rw=0.9,reps=2500,delta=0.0):
    rw_=noise_corr(rw); rb_=noise_corr(rb); g=[]
    for _ in range(reps):
        n=345; yy=(rng.random(n)<p).astype(int); zc=rng.standard_normal(n)
        def team(k,dd):
            zt=rng.standard_normal(n); E=np.sqrt(rb_)*zc[:,None]+np.sqrt(rw_-rb_)*zt[:,None]+np.sqrt(1-rw_)*rng.standard_normal((n,k)); S=dd*yy[:,None]+E
            return max(auc(yy,S[:,j]) for j in range(k))
        g.append(team(k2,d+delta)-team(k1,d))
    return np.array(g)
rows=[]
for rb in [0.8,0.9]:
    for k1,k2 in [(4,4),(4,6),(4,13),(1,13)]:
        g=sim_gap(k1,k2,rb); rows.append(dict(between_corr=rb,k_team=k1,k_rival=k2,E_gap=round(g.mean(),4),SD_gap=round(g.std(),4),P_ge_0018=round((g>=0.018).mean(),3),P_ge_0025=round((g>=0.025).mean(),3),P_ge_0028=round((g>=0.028).mean(),3),q95=round(np.quantile(g,0.95),4)))
print(pd.DataFrame(rows).to_string(index=False))
print('\n=== (e) implied real gap: observed gap minus median noise+selection gap (k1=4, k2=13), and the true delta that makes P(observed >= gap) = 50%')
for rb in [0.8,0.9]:
    g=sim_gap(4,13,rb); print(f'  between_corr {rb}: median noise+selection gap {np.median(g):.4f}, 90% interval [{np.quantile(g,0.05):.3f},{np.quantile(g,0.95):.3f}]; implied real delta for observed 0.018/0.025/0.028: {0.018-np.median(g):+.4f} / {0.025-np.median(g):+.4f} / {0.028-np.median(g):+.4f}')
# true-AUC shift needed: delta in d-units -> AUC units
for rb in [0.9]:
    for dAUC in [0.005,0.01,0.015,0.02]:
        dd=np.sqrt(2)*stats.norm.ppf(0.84+dAUC)-d; g=sim_gap(4,13,rb,delta=dd,reps=1500); print(f'  if rival TRUE AUC is higher by {dAUC}: P(observed best-of-13 minus best-of-4 >= 0.025) = {(g>=0.025).mean():.3f}; mean observed gap {g.mean():.4f}')
