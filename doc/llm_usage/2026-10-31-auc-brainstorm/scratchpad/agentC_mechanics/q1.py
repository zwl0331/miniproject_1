import pipe, traceback, time
F2 = pipe.F2_COLS
Lp_blocks = [pipe.block_date_count, pipe.make_block_date_te(3.0)]
days_idx = F2.index('days')
def mbf(n_extra=0, days=4095):
    v = [255]*len(F2) + [255]*n_extra; v[days_idx] = days; return v
jobs = [
 # A: histogram resolution on L
 ('A_L_mb4095_mdb3',  dict(model_fn=pipe.lgb_factory(max_bin=4095))),
 ('A_L_mb4095_mdb1',  dict(model_fn=pipe.lgb_factory(max_bin=4095, min_data_in_bin=1))),
 ('A_L_mb255_mdb1',   dict(model_fn=pipe.lgb_factory(min_data_in_bin=1))),
 ('A_L_mb1023_mdb3',  dict(model_fn=pipe.lgb_factory(max_bin=1023))),
 ('A_L_mbf_days4095_mdb1', dict(model_fn=pipe.lgb_factory(max_bin_by_feature=mbf(), min_data_in_bin=1))),
 # B: leaf size on L (default bins)
 ('B_L_mcs3',  dict(model_fn=pipe.lgb_factory(min_child_samples=3))),
 ('B_L_mcs5',  dict(model_fn=pipe.lgb_factory(min_child_samples=5))),
 ('B_L_mcs20', dict(model_fn=pipe.lgb_factory(min_child_samples=20))),
 # F: smoothing variants (L+ base: date_count + te variant)
 ('F_Lp_te_m1',      dict(model_fn=pipe.lgb_factory(), blocks=[pipe.block_date_count, pipe.make_block_date_te(1.0)])),
 ('F_Lp_te_m0.5',    dict(model_fn=pipe.lgb_factory(), blocks=[pipe.block_date_count, pipe.make_block_date_te(0.5)])),
 ('F_Lp_te_m0_cnt',  dict(model_fn=pipe.lgb_factory(), blocks=[pipe.block_date_count, pipe.make_block_date_te(0.0, with_count=True)])),
 ('F_Lp_te_m3_grp',  dict(model_fn=pipe.lgb_factory(), blocks=[pipe.block_date_count, pipe.make_block_date_te(3.0, per_group=True)])),
 # E: within-date NN features
 ('E_L_nn',   dict(model_fn=pipe.lgb_factory(), blocks=[pipe.block_date_nn])),
 ('E_Lp_nn',  dict(model_fn=pipe.lgb_factory(), blocks=Lp_blocks + [pipe.block_date_nn])),
]
for name, kw in jobs:
    try:
        pipe.evaluate(name, **kw)
    except Exception:
        print('FAILED', name); traceback.print_exc(); 
print('Q1 DONE', time.strftime('%H:%M:%S'), flush=True)
