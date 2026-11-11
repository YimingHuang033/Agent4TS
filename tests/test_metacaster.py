import copy
import json
from pathlib import Path
import numpy as np
from agent4ts.baselines.metacaster import support_tensor, synthetic_windows, sandbox_python
from agent4ts.transforms import Window,LabeledWindow,Normalizer


def fixture_window():
    w=Window(np.array([[1.,2.,3.]]),np.ones((1,3),dtype=np.int64),np.zeros((1,3)),np.array([0.,1.,2.]),np.array([0]),7,2.)
    return LabeledWindow(w,np.array([.5,1.]),np.array([0,0]),np.array([4.,5.]),np.array([True,False]))


def test_future_never_imputes_history_and_masks_survive():
    lw=fixture_window();norm=Normalizer('none')
    x,hg,fg=support_tensor([lw],norm,5,3,2.,1.)
    np.testing.assert_allclose(x[0,:5,0],[1.,1.5,2.,2.5,3.])
    changed=copy.deepcopy(lw);changed.y_true[:]=10000
    y,_,_=support_tensor([changed],norm,5,3,2.,1.)
    np.testing.assert_array_equal(x[:,:5],y[:,:5])
    restored=synthetic_windows(x,[lw],norm,hg,fg)[0]
    np.testing.assert_array_equal(restored.y_scored,lw.y_scored)
    np.testing.assert_array_equal(restored.window.mask,lw.window.mask)
    assert np.isnan(restored.y_true[1])


def test_execution_cannot_read_private_data_or_see_cuda(tmp_path):
    import sys
    if sys.platform!='linux':return
    result=sandbox_python("import os; assert not os.path.exists('/home/tim/Proj/ats'); assert not os.path.exists('/home/tim/.config/agent4ts/ark.env'); assert not os.path.exists('/mnt/data/ats_resources'); assert os.environ['CUDA_VISIBLE_DEVICES']==''; assert not any('KEY' in k for k in os.environ); print('isolated')",tmp_path)
    assert 'exit_code=0' in result and 'isolated' in result


def test_finalize_reuses_support_fitted_models(tmp_path,monkeypatch):
    from agent4ts import config
    from agent4ts.data import load_task
    from agent4ts.models.runners import DLinear
    from agent4ts.baselines.metacaster import run_metacaster
    cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3','gen_eval/metacaster_1111'])
    cfg['metacaster']['models']=['DLinear']
    cfg['metacaster']['support_size']=30
    bundle=load_task(cfg['tasks']['physiome_dupont'],cfg)
    original=DLinear.fit;seen=[]
    def fit(self,train,*args,**kwargs):
        seen.append([w.window.entity_id for w in train]);return original(self,train,*args,**kwargs)
    monkeypatch.setattr(DLinear,'fit',fit)
    result=run_metacaster(bundle,cfg,tmp_path,control=True)
    assert result['status']=='ok'
    assert len(seen)==1, 'confirm/test must not refit on the full training set'
    assert len(seen[0])==30
    assert set(seen[0]).issubset(bundle.spec.split_ids['train'])
    assert not set(seen[0]) & set(bundle.spec.split_ids['test'])


def test_generated_models_finalize_without_retraining_even_at_call_limit(tmp_path,monkeypatch):
    from agent4ts import config
    from agent4ts.data import load_task
    from agent4ts.models.runners import DLinear
    from agent4ts.baselines import metacaster
    cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3','gen_eval/metacaster_1111'])
    cfg['metacaster'].update(models=['DLinear'],support_size=30)
    bundle=load_task(cfg['tasks']['physiome_dupont'],cfg)
    def generation(samples,cfg,workspace,seed):
        return samples.copy(),[{}]*24
    monkeypatch.setattr(metacaster,'generate',generation)
    original=DLinear.fit;seen=[]
    def fit(self,train,*args,**kwargs):
        seen.append([w.window.entity_id for w in train]);return original(self,train,*args,**kwargs)
    monkeypatch.setattr(DLinear,'fit',fit)
    result=metacaster.run_metacaster(bundle,cfg,tmp_path)
    assert result['status']=='ok' and result['used']['llm_calls']==24
    assert len(seen)==1 and all(i<0 for i in seen[0])
