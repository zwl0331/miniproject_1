from common import *
from scipy import stats
from sklearn.metrics import roc_auc_score
tr,te,sp=load()
tr=tr.join(sp[['eligible','primary_group']]); el=tr[tr.eligible==1].reset_index(drop=True)
y=el.y.values
def rate(mask,name):
    m=np.asarray(mask); n=m.sum(); r=y[m].mean() if n else np.nan
    print(f'  {name:55s} n={n:5d} High={r:.3f}')
print('=== Q3: assembly / sources')
print('label rate by date precision:')
rate(el.date_len==10,'full YYYY-MM-DD'); rate(el.date_len==4,'YYYY only'); rate(el.date_len==7,'YYYY-MM')
print('  YYYY-only rows by era:'); print(el[el.date_len==4].groupby(pd.cut(el.year,[0,1989,1999,2009,2019,2030]))['y'].agg(['size','mean']).round(3).to_string())
print('  full-date rows by era:'); print(el[el.date_len==10].groupby(pd.cut(el.year,[0,1989,1999,2009,2019,2030]))['y'].agg(['size','mean']).round(3).to_string())
print('pre-1990 block, year counts:',el[el.year<1990].year.value_counts().sort_index().to_dict())
print('1990-1994 by year:',el[(el.year>=1990)&(el.year<1995)].groupby('year')['y'].agg(['size','mean']).round(2).T.to_dict())
# placeholder dates: Jan 1 and day==1
rate((el.month==1)&(el.day==1),'date is YYYY-01-01'); rate((el.day==1)&~((el.month==1)),'day==01, not January'); rate((el.day>1),'day>1')
print('  YYYY-01-01 by era:'); print(el[(el.month==1)&(el.day==1)].groupby(pd.cut(el.year,[0,1989,1999,2009,2019,2030]))['y'].agg(['size','mean']).round(3).to_string())
# audio differences pre-1990 vs 1990s vs rest
feat=['duration_ms','loudness','time_signature','energy','acousticness','instrumentalness','danceability','tempo','valence','speechiness','liveness']
era=pd.cut(el.year,[0,1989,1999,2009,2019,2030],labels=['<1990','1990s','2000s','2010s','2020+'])
print('\nfeature medians by era:'); print(el.groupby(era)[feat].median().round(3).to_string())
print('time_signature by era (row %):'); print(pd.crosstab(era,el.time_signature,normalize='index').round(3).to_string())
print('mode by era (share mode=1):',el.groupby(era)['mode'].mean().round(3).to_dict())
# within the 1990s/2000s: does anything separate High from Low more than in 2020+? univariate AUC per era
print('\nunivariate AUC of each feature by era (High vs Low):')
rows=[]
for e in ['1990s','2000s','2010s','2020+']:
    m=(era==e).values; r={'era':e,'n':m.sum()}
    for f in feat+['year','date_len','month','day','dow']:
        v=el[f].values[m]; ok=~np.isnan(v)
        r[f]=round(roc_auc_score(y[m][ok],v[ok]),3) if len(np.unique(y[m][ok]))>1 else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string())
# loudness rounding / duration rounding
print('\nduration_ms: last 3 digits ==000:',(el.duration_ms%1000==0).mean().round(4),' High rate among them:',y[(el.duration_ms%1000==0).values].mean().round(3),' n=',(el.duration_ms%1000==0).sum())
print('duration_ms: %1000 distribution top values:',(el.duration_ms%1000).value_counts().head(5).to_dict())
rate(el.duration_ms%1000==0,'duration ms multiple of 1000')
# many-song dates
dvc=el.date_str.value_counts(); cnt=el.date_str.map(dvc)
print('\nlabel rate by number of train rows sharing the exact date (full dates):')
for lo,hi in [(1,1),(2,2),(3,4),(5,9),(10,19),(20,999)]:
    m=((cnt>=lo)&(cnt<=hi)&(el.date_len==10)).values; rate(m,f'date count {lo}-{hi}')
# controlling for year: residual of label on year-era
era_rate=el.groupby(era)['y'].transform('mean').values
print('  mean(y - era rate) by date count:')
for lo,hi in [(1,1),(2,2),(3,4),(5,9),(10,19),(20,999)]:
    m=((cnt>=lo)&(cnt<=hi)&(el.date_len==10)).values; print(f'    {lo}-{hi}: {np.mean(y[m]-era_rate[m]):+.3f} (n={m.sum()})')
print('dates with >=10 rows:'); big=dvc[dvc>=10]; 
for d,c in big.items(): sub=el[el.date_str==d]; print(f'   {d}: n={c} High={sub.y.mean():.2f} groups={sub.primary_group.nunique()} dur_med={sub.duration_ms.median():.0f} loud_med={sub.loudness.median():.1f}')
# label rate of repeated-uri rows vs singletons (test-set shift)
uvc=el.uri.value_counts(); dup=el.uri.map(uvc)>1
rate(dup,'rows whose uri repeats in train'); rate(~dup,'rows with unique uri')
print('  by era, High rate dup vs unique:'); print(el.assign(dup=dup).groupby([era,'dup'])['y'].agg(['size','mean']).round(3).unstack().to_string())
# conflicting-label groups
gy=el.groupby('primary_group')['y'].agg(['size','mean','nunique']); conf=gy[(gy['size']>1)&(gy['nunique']>1)]
print('\nmulti-row groups:',(gy['size']>1).sum(),' conflicting:',len(conf), ' sizes of conflicting:',conf['size'].value_counts().to_dict())
cg=el[el.primary_group.isin(conf.index)]; print('  conflicting-group rows by era:',cg.groupby(era)['y'].size().to_dict(),' vs multi-row consistent groups by era:',el[el.primary_group.isin(gy[(gy['size']>1)&(gy['nunique']==1)].index)].groupby(era)['y'].size().to_dict())
print('  are conflicting rows identical in audio AND date?', cg.groupby('primary_group').apply(lambda d: (d[AUDIO+['date_str']].nunique()<=1).all()).mean().round(3))
print('  conflicting groups: date differs within group:', cg.groupby('primary_group')['date_str'].nunique().gt(1).sum())
# mode/key by era label
print('\n=== Q7 misc')
print('release weekday distribution by era (full dates), row %:'); print(pd.crosstab(era[el.date_len==10],el.dow[el.date_len==10],normalize='index').round(3).to_string())
print('High rate by weekday within era:'); print(el[el.date_len==10].groupby([era,'dow'])['y'].mean().round(2).unstack().to_string())
print('High rate by weekday, 2015+ only:',el[(el.year>=2015)].groupby('dow')['y'].agg(['size','mean']).round(3).T.to_dict())
print('High rate by mode within era:'); print(el.groupby([era,'mode'])['y'].agg(['size','mean']).round(3).unstack().to_string())
print('High rate by time_signature:'); print(el.groupby('time_signature')['y'].agg(['size','mean']).round(3).to_string())
print('High rate by key (all):',el.groupby('key')['y'].mean().round(2).to_dict())
# loudness clustering: decimals
print('loudness decimals (digits after point) distribution:',el.loudness.astype(str).str.split('.').str[1].str.len().value_counts().to_dict())
print('tempo: fraction in [60,90):',((el.tempo>=60)&(el.tempo<90)).mean().round(3),' High rate by tempo band:',el.groupby(pd.cut(el.tempo,[0,70,90,110,130,150,170,250]))['y'].agg(['size','mean']).round(3).T.to_dict())
# year-month day structure
print('High rate by month (2015+):',el[el.year>=2015].groupby('month')['y'].mean().round(2).to_dict())
# days-since-release monotonic? within 2020+, by year
print('2018-2025 by year:',el[el.year>=2018].groupby('year')['y'].agg(['size','mean']).round(3).T.to_dict())
# Q5 follow-up: duplicate-removal hypothesis simulation
print('\n=== Q5 follow-up: can "removed rows = test rows whose uri is in train" explain 103?')
mult=uvc.value_counts().sort_index(); print('train uri multiplicities:',mult.to_dict())
rng=np.random.default_rng(1)
# full data: N=4830 rows; unknown number G2 of uri groups with size 2, G3 size 3 ...; fit by simulation to observed train (203 repeated uris/424 rows), test-test (11 uris/22 rows)
def sim(G2,G3,G4=0,reps=200):
    out=[]
    for _ in range(reps):
        sizes=[2]*G2+[3]*G3+[4]*G4; nd=sum(sizes); n1=4830-nd
        uri=np.concatenate([np.repeat(np.arange(len(sizes)),sizes),np.arange(len(sizes),len(sizes)+n1)])
        perm=rng.permutation(4830); test=np.zeros(4830,bool); test[perm[:966]]=True
        tru=set(uri[~test]); tr_rep=pd.Series(uri[~test]).value_counts(); te_u=uri[test]
        removed=np.isin(te_u,list(tru)).sum(); te_rep=pd.Series(te_u[~np.isin(te_u,list(tru))]).value_counts()
        out.append((int((tr_rep>1).sum()),int(tr_rep[tr_rep>1].sum()),int(removed),int((te_rep>1).sum())))
    return np.array(out).mean(0).round(1)
for G2,G3 in [(250,30),(270,25),(300,20),(280,30)]:
    print(f'  G2={G2} G3={G3}: [train repeated uris, train rows in them, test rows removed, test repeated uris] =',sim(G2,G3))
print('  observed: [203, 424, 103, 11]')
