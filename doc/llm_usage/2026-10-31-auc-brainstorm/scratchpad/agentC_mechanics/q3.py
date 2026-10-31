import pipe, traceback, time
Lp_blocks = [pipe.block_date_count, pipe.make_block_date_te(3.0)]
jobs = [
 ('B_Lp_mcs5',       dict(model_fn=pipe.lgb_factory(min_child_samples=5), blocks=Lp_blocks)),
 ('B_Lp_mcs3',       dict(model_fn=pipe.lgb_factory(min_child_samples=3), blocks=Lp_blocks)),
 ('B_Lp_mcs5_seed1', dict(model_fn=pipe.lgb_factory(min_child_samples=5), blocks=Lp_blocks, model_seeds=(1,))),
 ('B_Lp_mcs5_seed2', dict(model_fn=pipe.lgb_factory(min_child_samples=5), blocks=Lp_blocks, model_seeds=(2,))),
 ('B_L_mcs5_seed1',  dict(model_fn=pipe.lgb_factory(min_child_samples=5), model_seeds=(1,))),
 ('B_L_mcs5_seed2',  dict(model_fn=pipe.lgb_factory(min_child_samples=5), model_seeds=(2,))),
 ('G_L_seed1',       dict(model_fn=pipe.lgb_factory(), model_seeds=(1,))),
 ('G_L_seed2',       dict(model_fn=pipe.lgb_factory(), model_seeds=(2,))),
]
for name, kw in jobs:
    try:
        pipe.evaluate(name, **kw)
    except Exception:
        print('FAILED', name); traceback.print_exc()
print('Q3 DONE', time.strftime('%H:%M:%S'), flush=True)
