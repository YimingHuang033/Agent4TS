"""V3 fairness, common API recovery and raw observation contracts."""
import copy,json
from pathlib import Path
from agent4ts.action_space import ActionSpace
from agent4ts.schemas import PipelineSpec,DataView,Action
from agent4ts.controller import Controller

def test_canonical_default_is_not_new_candidate():
    f0=PipelineSpec('a',DataView(),'DLinear',{},0);space=ActionSpace('model_only',f0)
    other=copy.deepcopy(f0);other.model_config={'epochs':200}
    assert space.key(f0)==space.key(other)
    other.model_config={'epochs':100};assert space.key(f0)!=space.key(other)

def test_data_only_forbids_model_and_options():
    space=ActionSpace('data_only',PipelineSpec('a',DataView(),'APN',{},0))
    assert not space.permits(Action(op='SELECT_MODEL',value='DLinear'))
    assert not space.permits(Action(op='SET_MODEL_OPTION',target_field='epochs',value=100))
    assert space.permits(Action(op='SET_TRANSFORM',target_field='history_fraction',value=.5))

def test_model_only_freezes_view():
    space=ActionSpace('model_only',PipelineSpec('a',DataView(),'APN',{},0))
    assert not space.permits(Action(op='SET_TRANSFORM',target_field='normalize',value='none'))
    assert space.permits(Action(op='SELECT_MODEL',value='DLinear'))
    assert ActionSpace('model_only',space.f0).hash==space.hash

def test_common_transport_retry_once():
    ctrl=Controller({'llm':{'transport':'mock'}},Path('/tmp'))
    calls=[]
    def call(messages):calls.append(1);return '' if len(calls)==1 else '{"ok":true}'
    ctrl._call=call
    assert ctrl.request([],json.loads)=={'ok':True};assert len(calls)==2
    calls.clear();ctrl._call=lambda messages:calls.append(1) or 'bad json'
    try:ctrl.request([],json.loads)
    except RuntimeError:pass
    else:raise AssertionError('Invalid reply silently accepted')
    assert len(calls)==2


def test_grid_uses_unweighted_mean_and_retains_cutoff():
    import numpy as np
    from agent4ts.schemas import EventTable
    from agent4ts.transforms import _events_to_window
    t=np.array([.1,.2,.3,2.]);values=np.array([1.,3.,8.,16.])
    events=EventTable(np.zeros(4,dtype=int),t,t,np.zeros(4,dtype=int),values)
    window=_events_to_window(events,np.array([0]),2.,DataView(grid_mode='grid',grid_width=1.),None,{0:2.})
    np.testing.assert_allclose(window.x,[[4.,16.]])
    assert window.mask.sum()==2

import pytest
@pytest.mark.parametrize('ablation',['no_vision','no_semantics','one_shot'])
def test_matched_ablation_boundary(ablation,tmp_path,monkeypatch):
    from agent4ts import config
    from agent4ts.data import load_task
    from agent4ts.baselines import greedy_search
    cfg=config.load_config('smoke/quick',extra=['gen_eval/expand_v3'])
    cfg['tasks']['physiome_dupont']['n_entities']=10
    bundle=load_task(cfg['tasks']['physiome_dupont'],cfg);bundle.spec.f0_model='LastValue'
    observed=[]
    class Probe:
        def __init__(self,*args):pass
        def propose(self,obs):
            observed.append(obs)
            return Action(op='INSPECT' if ablation=='one_shot' else 'STOP')
    monkeypatch.setattr(greedy_search,'Controller',Probe)
    greedy_search.run_matched(bundle,cfg,tmp_path,policy='agent',ablation=ablation)
    assert len(observed)==1
    obs=observed[0]
    if ablation=='no_semantics':assert obs.dataset_card=={}
    if ablation=='no_vision':
        assert obs.plot_paths==[]
        messages=Controller(cfg,tmp_path).build_messages(obs)
        assert not any(c.get('type')=='image_url' for c in messages[1]['content'])
