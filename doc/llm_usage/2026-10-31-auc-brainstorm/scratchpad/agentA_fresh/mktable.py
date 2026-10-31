import pandas as pd
r=pd.read_csv('results.csv').drop_duplicates('name',keep='last')
base=r[r.name=='lgb_base'].iloc[0]
r['d_mean']=r.mean15-base.mean15; r['d155']=r.s155-base.s155; r['d2026']=r.s2026-base.s2026; r['d1029']=r.s1029-base.s1029; r['d_single']=r.single15-base.single15
cols=['name','mean15','s155','s2026','s1029','single15','d_mean','d155','d2026','d1029','d_single','secs']
out=['| experiment | mean15 | seed155 | seed2026 | seed1029 | singleton15 | d mean | d 155 | d 2026 | d 1029 | d single | secs |','|---|---|---|---|---|---|---|---|---|---|---|---|']
for _,x in r.iterrows():
    out.append('| '+' | '.join([x['name']]+[f'{x[c]:.4f}' for c in cols[1:6]]+[f'{x[c]:+.4f}' for c in cols[6:11]]+[f'{x.secs:.0f}'])+' |')
open('table.md','w').write('\n'.join(out)+'\n')
b=pd.read_csv('blend_results.csv'); bb=b[b.blend=="{'lgb': 1}"].iloc[0]
out=['| rank-average blend (weights) | mean15 | seed155 | seed2026 | seed1029 | singleton15 | d mean vs lgb alone | d single |','|---|---|---|---|---|---|---|---|']
for _,x in b.iterrows():
    out.append(f"| {x.blend} | {x.mean15:.4f} | {x.s155:.4f} | {x.s2026:.4f} | {x.s1029:.4f} | {x.single15:.4f} | {x.mean15-bb.mean15:+.4f} | {x.single15-bb.single15:+.4f} |")
open('table_blend.md','w').write('\n'.join(out)+'\n')
