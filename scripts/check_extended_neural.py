"""Focused checks for feature transfer, mixed inputs and group-safe retrieval."""
import copy
import json
import subprocess
import sys
import unittest
import numpy as np
from scripts.check_neural_adapter import AdvancedNeural as A
from songpop.phase2 import raw_neural as N, raw_recipes as R, tabm as T
from songpop.phase2.encoders import inner_group_splits


class Extended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        A.setUpClass()
    @classmethod
    def tearDownClass(cls):
        N.Worker.close_all()

    def test_extended_features_stopping_isolation(self):
        recipe = R.make_recipe('tabm',representation='ple',device='cuda',layout='mixed_neural_v1',
            features={'year_relative':True,'year_standardize':True,'date_counts':True},
            blocks=['date_count','date_te3'],stopping_metric='auc',pairwise_weight=.5,
            params={'max_epochs':3,'patience':2,'d_block':32,'n_blocks':1})
        fit,valid=T.inner_split(A.y,A.groups,5,T.stage_seeds(A.context,0)['inner_split'])
        diag={'stop_split':(fit,valid),'stop_encoder_splits':inner_group_splits(A.y[fit],A.groups[fit],A.context.inner_seed('date_te'))}
        req=dict(op='trajectory',recipe=recipe,rows=A.rows,y=A.y,groups=A.groups,context=A.context,diagnostics=diag,epochs=3,seed=0)
        worker=N.Worker.for_device('cuda'); before=worker.call(req)
        labels=A.y.copy(); labels[valid]=1-labels[valid]
        changed=A.rows.copy(); changed.iloc[valid,changed.columns.get_loc('loudness')]+=1000
        after=worker.call(dict(req,rows=changed,y=labels))
        for k in ['trajectory','feature_state_sha256','model_spec_sha256','preprocessing_sha256','Zs_sha256']:
            self.assertEqual(before[k],after[k],k)
        self.assertNotEqual(before['Zv_sha256'],after['Zv_sha256'])
        changed=A.rows.copy(); changed.iloc[fit[0],changed.columns.get_loc('loudness')]+=1000
        positive=worker.call(dict(req,rows=changed))
        self.assertNotEqual(before['model_spec_sha256'],positive['model_spec_sha256'])

    def test_mixed_unknown_categories_and_bins(self):
        recipe=R.make_recipe('mlp',representation='ple',device='cuda',layout='mixed_neural_v1',
            params={'max_epochs':2,'patience':2,'d_block':32,'n_blocks':1})
        fitted=N.fit_raw(recipe,A.rows,A.y,A.groups,A.context)
        self.assertEqual(len(fitted.model_spec['bins']),fitted.model_spec['n_cont'])
        query=A.rows.iloc[:5].copy(); query['key']=999
        Z=fitted.model_inputs(query)
        catpos=fitted.model_spec['n_cont']+fitted.model_spec['n_flags']+fitted.prep.cat_.index('key')
        self.assertTrue((Z[:,catpos]==0).all())
        self.assertTrue(np.isfinite(fitted.predict_high(query)).all())

    def test_retrieval_group_mask_and_no_supervised_features(self):
        with self.assertRaises(ValueError):
            R.make_recipe('retrieval',blocks=['date_te3'])
        code='''
import numpy as np
from songpop.phase2.neural_backend import configure
from songpop.phase2.neural_architectures import specification,build
from songpop.phase2.raw_recipes import make_recipe
torch=configure('cpu')
recipe=make_recipe('retrieval',device='cpu',params={'context_size':4,'d_block':16,'n_blocks':1})
Z=np.random.default_rng(0).normal(size=(20,5)).astype('float32')
net=build(torch,recipe,specification(recipe,Z),155)
x=torch.from_numpy(Z); y=torch.arange(20)%2; g=torch.arange(20)//2
net.set_memory(x,y,g); net.eval()
with torch.no_grad():
 before=net(x[:4],g[:4]); ix=net.last_indices_
 assert not (net.memory_groups[ix]==g[:4,None]).any()
 # Fixed weights: changing only query-group reference labels must have no
 # effect on that group's prediction; other-group labels must matter.
 original=net(x[:2],g[:2]); net.memory_y[:2]=1-net.memory_y[:2]
 assert torch.equal(original,net(x[:2],g[:2]))
 net.memory_y[2:]=1-net.memory_y[2:]
 assert not torch.equal(original,net(x[:2],g[:2]))
print('PASS retrieval group exclusion and label positive control')
'''
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=120)
        self.assertEqual(r.returncode,0,r.stderr)
        print(r.stdout.strip())


if __name__=='__main__':
    unittest.main(defaultTest='Extended',verbosity=2)
