"""CPU validation of actual new data, frozen prediction and shared search space."""
import copy,json,pickle
from pathlib import Path
import numpy as np
from agent4ts import config
from agent4ts.data import load_task
from agent4ts.harness import Harness
from agent4ts.schemas import PipelineSpec,DataView
from agent4ts.action_space import ActionSpace
from agent4ts.search import initial_spec
from agent4ts.baselines.greedy_search import run_matched
from agent4ts.tests_checks import run_checks
from agent4ts.models.registry import runnable_ids

cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3'])
cfg['resources']['upstream_training_epochs']=1
cfg['llm']['transport']='mock'; cfg['budget']['trials']=4
root=Path(cfg['expansion']['audit_root']);rows=[]
for task in cfg['expansion']['new_tasks']:
    bundle=load_task({'backend':task.removesuffix('_forecast')},cfg)
    assert run_checks(bundle,cfg)['status']=='ok'
    # Tiny train/search/confirm/test subset for model integration only. Full-data audit is separate.
    bundle=copy.deepcopy(bundle)
    for split,ids in bundle.spec.split_ids.items():bundle.spec.split_ids[split]=ids[:3]
    ids=sum(bundle.spec.split_ids.values(),[])
    bundle.history=bundle.history.subset(np.isin(bundle.history.entity_id,ids))
    bundle.targets=bundle.targets.subset(np.isin(bundle.targets.entity_id,ids))
    bundle.entity_anchors={i:a for i,a in bundle.entity_anchors.items() if i in ids}
    for model in runnable_ids():
        out=root/task/'models'/model;h=Harness(bundle,cfg,out);p=PipelineSpec('debug',DataView(),model,{},0)
        fb=h.run_trial(p);assert fb.status=='ok',(task,model,fb.error)
        payload=h.export(p,out/'export');frozen=pickle.loads((out/'export'/payload['model_artifact']).read_bytes())
        lw=h.windows('search',p.data_view)[0];before=frozen.predict(lw);other=copy.deepcopy(lw)
        other.y_true[:]=999999;other.y_scored[:]=False
        np.testing.assert_allclose(before,frozen.predict(other));assert np.isfinite(before).all()
        row={'task':task,'model':model,'status':fb.status,'normalized_mae':fb.metrics['normalized_mae'],
             'prediction_std':float(np.std(before)),
             'prediction_std_by_variable':{bundle.var_name(int(v)):float(np.std(before[lw.query_var_ids==v])) for v in np.unique(lw.query_var_ids)},'target_std':float(np.std(lw.y_true)),
             'training_diagnostics':getattr(frozen,'training_diagnostics',None),'reload_label_independence':'passed'}
        rows.append(row);print(json.dumps(row),flush=True)
    for policy in ['random','greedy']:
        search=run_matched(bundle,cfg,root/task/policy,seed=0,policy=policy)
        assert search['used']['trials']==4,search['used']
        assert len({e['fingerprint'] for e in search['archive']})==len(search['archive'])
        assert not any(json.loads(line).get('split') in ('test','confirm') for line in (root/task/policy/'trace.jsonl').read_text().splitlines())
        rows.append({'task':task,'policy':policy,'status':'ok','used':search['used']})
    space=ActionSpace('data_only',initial_spec(bundle));h=Harness(bundle,cfg,root/task/'domain')
    for key,action,p in space.neighbors(initial_spec(bundle),h,set()):assert p.model_id=='APN' and p.model_config=={}
(root/'debug_passed.json').write_text(json.dumps({'status':'passed','scope':'CPU, actual data, tiny per-split integration subsets; no formal scores','results':rows},indent=2))
