from common import *
from scipy import stats
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(42)
print('=== Q6 (a): Hanley-McNeil SE of a single AUC, A=0.84, positive rate 0.35')
def hm_se(A,n,p=0.35):
    n1=round(n*p); n0=n-n1; Q1=A/(2-A); Q2=2*A*A/(1+A)
    return np.sqrt((A*(1-A)+(n1-1)*(Q1-A*A)+(n0-1)*(Q2-A*A))/(n1*n0))
for frac,n in [(0.2,173),(0.3,259),(0.4,345),(0.5,432),(0.6,518),(1.0,863)]:
    print(f'  fraction {frac:.0%} n={n:3d}: SE={hm_se(0.84,n):.4f}  (95% half-width {1.96*hm_se(0.84,n):.3f})')
# empirical check from OOF preds of the probe (model E, seed 155): bootstrap SE at n=345 by subsampling
Z=np.load('oof_probe.npz'); y=Z['y']
pA=np.mean([Z[f'A_base_{s}'] for s in ['155','2026','1029']],axis=0); pE=np.mean([Z[f'E_wmean_{s}'] for s in ['155','2026','1029']],axis=0)
pB=np.mean([Z[f'B_te_{s}'] for s in ['155','2026','1029']],axis=0)
print(f'  OOF (seed-averaged) AUC: A={roc_auc_score(y,pA):.4f} B={roc_auc_score(y,pB):.4f} E={roc_auc_score(y,pE):.4f}; corr(A,E)={np.corrcoef(pA,pE)[0,1]:.3f} corr(A,B)={np.corrcoef(pA,pB)[0,1]:.3f}')
for n in [345,518,863]:
    a=[];d=[]
    for _ in range(3000):
        ii=rng.choice(len(y),n,replace=False)
        if y[ii].sum()<5: continue
        a.append(roc_auc_score(y[ii],pA[ii])); d.append(roc_auc_score(y[ii],pE[ii])-roc_auc_score(y[ii],pA[ii]))
    print(f'  empirical subsample n={n}: SE(AUC_A)={np.std(a):.4f}; SE(AUC_E-AUC_A)={np.std(d):.4f} (corr 0.98)')
print('\n=== Q6 (b): simulated SE of the paired AUC difference of two models with equal true AUC 0.84')
p=0.35; d=np.sqrt(2)*stats.norm.ppf(0.84); vy=d*d*p*(1-p)
def sim_pair(n,rho_pred,reps=4000):
    r=rho_pred*(vy+1)-vy  # noise correlation giving prediction correlation rho_pred
    out=[]
    for _ in range(reps):
        yy=(rng.random(n)<p).astype(int)
        if yy.sum()<2 or yy.sum()>n-2: continue
        e1=rng.standard_normal(n); e2=r*e1+np.sqrt(1-r*r)*rng.standard_normal(n)
        s1=d*yy+e1; s2=d*yy+e2
        out.append((roc_auc_score(yy,s1),roc_auc_score(yy,s2)))
    out=np.array(out); return out[:,0].std(), (out[:,1]-out[:,0]).std(), np.corrcoef(out[:,0],out[:,1])[0,1]
rows=[]
for n in [345,518,863]:
    for rho in [0.8,0.9,0.97]:
        se1,sed,ra=sim_pair(n,rho); rows.append(dict(n=n,pred_corr=rho,SE_single=round(se1,4),SE_diff=round(sed,4),AUC_corr=round(ra,3),gap_95=round(1.96*sed,3)))
T=pd.DataFrame(rows); print(T.to_string(index=False))
print('\n=== Q6 (c): winner\'s curse, n=345 public rows, k submissions with identical true AUC 0.84, pairwise prediction correlation rho')
def sim_max(n,rho_pred,k,reps=3000):
    r=rho_pred*(vy+1)-vy
    out=[]
    for _ in range(reps):
        yy=(rng.random(n)<p).astype(int)
        z=rng.standard_normal(n); E=np.sqrt(r)*z[:,None]+np.sqrt(1-r)*rng.standard_normal((n,k))
        S=d*yy[:,None]+E
        aucs=[roc_auc_score(yy,S[:,j]) for j in range(k)]
        out.append((max(aucs),aucs[0]))
    out=np.array(out); return out[:,0].mean()-out[:,1].mean(), out[:,0].std(), out[:,0].mean()
rows=[]
for rho in [0.8,0.9]:
    for k in [1,4,6,13,24]:
        infl,sd,mx=sim_max(345,rho,k); rows.append(dict(pred_corr=rho,k=k,E_max_minus_true=round(infl,4),SD_of_max=round(sd,4)))
T2=pd.DataFrame(rows); print(T2.to_string(index=False))
print('\n=== Q6 (d): gap between team X (k1 entries, reports its best) and team Y (k2 entries, best) when TRUE AUCs are equal; n=345')
def sim_gap(n,rho_within,rho_between,k1,k2,reps=3000):
    # all models share the y signal; within-team noise correlation r_w, between-team r_b (r_b<=r_w)
    rw=rho_within*(vy+1)-vy; rb=rho_between*(vy+1)-vy
    gaps=[]
    for _ in range(reps):
        yy=(rng.random(n)<p).astype(int)
        zc=rng.standard_normal(n)  # common to all
        def team(k):
            zt=rng.standard_normal(n); E=np.sqrt(rb)*zc[:,None]+np.sqrt(rw-rb)*zt[:,None]+np.sqrt(1-rw)*rng.standard_normal((n,k))
            S=d*yy[:,None]+E; return max(roc_auc_score(yy,S[:,j]) for j in range(k))
        gaps.append(team(k2)-team(k1))
    g=np.array(gaps); return g
rows=[]
for rb in [0.8,0.9]:
    for k1,k2 in [(4,4),(4,6),(4,13),(1,13)]:
        g=sim_gap(345,0.9,rb,k1,k2)
        rows.append(dict(between_corr=rb,k_team=k1,k_rival=k2,E_gap=round(g.mean(),4),SD_gap=round(g.std(),4),P_gap_ge_0018=round((g>=0.018).mean(),3),P_gap_ge_0025=round((g>=0.025).mean(),3),P_gap_ge_0028=round((g>=0.028).mean(),3),q95=round(np.quantile(g,0.95),4)))
T3=pd.DataFrame(rows); print(T3.to_string(index=False))
print('\n=== Q6 (e): if the true gap is delta, expected observed best-of-13 minus best-of-4 gap = delta + E_gap above; solve for the delta that makes an observed 0.025 gap the median outcome')
for rb in [0.8,0.9]:
    g=sim_gap(345,0.9,rb,4,13); print(f'  between_corr {rb}: median noise+selection gap {np.median(g):.4f}; implied real delta for observed 0.025: {0.025-np.median(g):.4f}; for 0.018: {0.018-np.median(g):.4f}; for 0.028: {0.028-np.median(g):.4f}; 90% interval of noise gap [{np.quantile(g,0.05):.3f},{np.quantile(g,0.95):.3f}]')
