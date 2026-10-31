import pipe, traceback, time
F2 = pipe.F2_COLS
Lp_blocks = [pipe.block_date_count, pipe.make_block_date_te(3.0)]
cat = pipe.lgb_cat_hook
jobs = [
 ('C_Lp_cat_cs10_mdpg5', dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=[255]*(len(F2)+2)+[4095], cat_smooth=10, min_data_per_group=5), blocks=Lp_blocks+[pipe.block_date_cat], fit_hook=cat)),
 ('C_L_cat_cs30_mdpg5',  dict(model_fn=pipe.lgb_factory(min_data_in_bin=1, max_bin_by_feature=[255]*len(F2)+[4095], cat_smooth=30, min_data_per_group=5), blocks=[pipe.block_date_cat], fit_hook=cat)),
]
for name, kw in jobs:
    try:
        pipe.evaluate(name, **kw)
    except Exception:
        print('FAILED', name); traceback.print_exc()
print('Q4 DONE', time.strftime('%H:%M:%S'), flush=True)
