"""Live Ark API integration, CPU-only tiny data subset (not formal scores)."""
import copy,json,os
from pathlib import Path
import numpy as np
from agent4ts import config
from agent4ts.data import load_task
from agent4ts.baselines.greedy_search import run_matched
from agent4ts.baselines.tsci_adapted import run_tsci_adapted
from agent4ts.search import finalize
from agent4ts.cli import pipeline_from_dict
cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3'])
cfg['budget']['trials']=4;cfg['budget']['llm_calls']=6;cfg['resources']['upstream_training_epochs']=1
assert cfg['llm']['provider']=='ark' and cfg['llm']['transport']=='openai'
root=Path(cfg['expansion']['audit_root'])/'api';rows=[]
for task in cfg['expansion']['new_tasks']:
    bundle=load_task({'backend':task.removesuffix('_forecast')},cfg)
    for split,ids in bundle.spec.split_ids.items():bundle.spec.split_ids[split]=ids[:3]
    ids=sum(bundle.spec.split_ids.values(),[])
    bundle.history=bundle.history.subset(np.isin(bundle.history.entity_id,ids));bundle.targets=bundle.targets.subset(np.isin(bundle.targets.entity_id,ids))
    bundle.entity_anchors={i:a for i,a in bundle.entity_anchors.items() if i in ids}
    for policy in ['agent','tsci']:
        out=root/task/policy
        result=run_matched(bundle,cfg,out,policy='agent',ablation='one_shot') if policy=='agent' else run_tsci_adapted(bundle,cfg,out)
        trace=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
        usages=[s for s in trace if s.get('event')=='llm_usage']
        assert usages,'No actual API response'
        assert all((s.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0)==0 for s in usages),'Ark thinking disable not honored'
        assert 'unparseable' not in result.get('stop_reason','')
        successful=sorted([s for s in result['archive'] if s['status']=='ok'],key=lambda s:s['metrics']['normalized_mae'])
        final=finalize(bundle,cfg,out,[pipeline_from_dict(s['pipeline']) for s in successful[:3]])
        assert final['status']=='ok',final
        row={'task':task,'policy':policy,'status':'ok','used':result['used'],'finalize':final['status'],'llm_model':cfg['llm']['model']}
        rows.append(row);print(json.dumps(row),flush=True)
(root/'report.json').write_text(json.dumps(rows,indent=2))
