"""Focused validity and held-out isolation checks for the bounded screen."""
import copy
import subprocess
import sys
import unittest
import numpy as np
from scripts.check_neural_adapter import AdvancedNeural as A
from songpop.phase2 import raw_neural as N, raw_recipes as R, tabm as T


class Augmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        A.setUpClass()

    @classmethod
    def tearDownClass(cls):
        N.Worker.close_all()

    def test_invalid_routes_rejected(self):
        aug = dict(kind="mixup", alpha=.1, probability=.5)
        for family, kwargs in [("retrieval", {}), ("mlp", {"pairwise_weight": .5})]:
            with self.assertRaises(ValueError):
                R.make_recipe(family, layout="mixed_neural_v1", augmentation=aug, **kwargs)
        with self.assertRaises(ValueError):
            R.make_recipe("mlp", augmentation=aug)

    def test_batch_validity_and_positive_controls(self):
        code = '''
import numpy as np
from songpop.phase2.neural_backend import configure
from songpop.phase2.neural_augmentation import Augmenter
torch=configure('cpu')
y=np.tile([0.,1.],20).astype('float32')
Z=np.column_stack([y, np.linspace(-1,1,40), np.repeat([0.,1.],20), np.repeat([1.,2.],20)]).astype('float32')
spec=dict(n_cont=2,n_flags=1,augmentation_audio=[0,1],augmentation_missing={'1':2})
x=torch.from_numpy(Z.copy()); labels=torch.from_numpy(y); ix=np.arange(40)
noise=dict(kind='noise',sigma=.2,probability=1.)
a=Augmenter(noise,spec,Z,123); noisy,target=a.batch(torch,x,labels,ix)
again,_=Augmenter(noise,spec,Z,123).batch(torch,x,labels,ix)
assert torch.equal(noisy,again) and torch.equal(target,labels)
assert torch.equal(noisy[:,2:],x[:,2:]) and torch.equal(noisy[20:,1],x[20:,1])
assert not torch.equal(noisy[:20,:2],x[:20,:2])
assert torch.equal(x,torch.from_numpy(Z))
mix=dict(kind='mixup',alpha=.4,probability=1.)
mixed,target=Augmenter(mix,spec,Z,123).batch(torch,x,labels,ix)
assert torch.equal(mixed[:,2:],x[:,2:])
assert torch.allclose(mixed[:,0],target) # feature/label pairing, not detached random labels
assert ((target>0)&(target<1)).any()
assert torch.equal(x,torch.from_numpy(Z))
print('PASS unchanged categories/flags/missing cells, paired labels, deterministic seed, input immutability')
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        print(result.stdout.strip())

    def test_stopping_isolation_and_gpu_smoke(self):
        recipe=R.make_recipe('mlp',device='cuda',layout='mixed_neural_v1',blocks=['date_count'],
            augmentation=dict(kind='mixup',alpha=.1,probability=.5),
            params=dict(max_epochs=3,patience=2,d_block=32,n_blocks=1))
        fit,valid=T.inner_split(A.y,A.groups,5,T.stage_seeds(A.context,0)['inner_split'])
        req=dict(op='trajectory',recipe=recipe,rows=A.rows,y=A.y,groups=A.groups,
                 context=A.context,diagnostics={'stop_split':(fit,valid)},epochs=3,seed=0)
        worker=N.Worker.for_device('cuda'); before=worker.call(req)
        labels=A.y.copy();labels[valid]=1-labels[valid]
        changed=A.rows.copy();changed.iloc[valid,changed.columns.get_loc('loudness')]+=1000
        after=worker.call(dict(req,rows=changed,y=labels))
        for key in ['trajectory','feature_state_sha256','model_spec_sha256','preprocessing_sha256','Zs_sha256']:
            self.assertEqual(before[key],after[key],key)
        self.assertNotEqual(before['Zv_sha256'],after['Zv_sha256'])
        labels=A.y.copy();labels[fit]=1-labels[fit]
        positive=worker.call(dict(req,y=labels))
        self.assertNotEqual(before['trajectory'],positive['trajectory'])
        for family,aug in [('tabm',dict(kind='mixup',alpha=.1,probability=.5)),
                           ('tabm',dict(kind='noise',sigma=.02,probability=.5)),
                           ('retrieval',dict(kind='noise',sigma=.02,probability=.5))]:
            r=R.make_recipe(family,device='cuda',layout='mixed_neural_v1',blocks=['date_count'],
                augmentation=aug,params=dict(max_epochs=2,patience=2,d_block=32,n_blocks=1))
            fitted=N.fit_raw(r,A.rows,A.y,A.groups,A.context)
            query=A.rows.iloc[:5]; original=fitted.model_inputs(query).copy()
            p=fitted.predict_high(query)
            self.assertTrue(np.isfinite(p).all() and ((p>=0)&(p<=1)).all())
            np.testing.assert_array_equal(original,fitted.model_inputs(query))
            print('PASS GPU',family,aug['kind'])


if __name__=='__main__':
    unittest.main(defaultTest='Augmentation',verbosity=2)
