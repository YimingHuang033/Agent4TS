"""Verify pinned resources; audit raw protocols and freeze task specs on CPU."""
import json,os
from pathlib import Path
import numpy as np
from agent4ts import config
from agent4ts.data import load_task
from agent4ts.data.raw_forecast import digest
from agent4ts.tests_checks import run_checks

cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3'])
root=Path(cfg['expansion']['audit_root']);root.mkdir(parents=True,exist_ok=True)
raw=Path(cfg['expansion']['raw_root']);snapshots={}
for name,meta in cfg['expansion']['snapshots'].items():
    p=raw/meta['relative_path']
    assert p.stat().st_size==meta['size'] and digest(p)==meta['sha256'],f'Pinned snapshot mismatch: {p}'
    import torch
    records=torch.load(p,map_location='cpu',weights_only=False)
    lengths=[len(r[1]) for r in records]
    snapshots[name]={**meta,'verified':True,'record_count':len(records),'length_range':[min(lengths),max(lengths)],
        'shapes':[list(r[2].shape) for r in records[:2]],'status':'rejected_for_formal_forecast',
        'reason':('Global normalization before splitting in GRU-ODE-Bayes process.py; absent scale and station mapping. Cannot recover raw-unit forecasts.' if name=='ushcn' else 'Only two anonymous records / four channels; exact column lineage and subject/session provenance not recoverable from snapshot. Use official raw data.')}
(root/'snapshot_audit.json').write_text(json.dumps(snapshots,indent=2))
lock_path=Path(cfg['paths']['resource_lock']);lock=json.loads(lock_path.read_text())
lock['expand_1109']={'snapshots':snapshots,'raw_files':{}}
for task in cfg['expansion']['new_tasks']:
    backend=task.removesuffix('_forecast');bundle=load_task({'backend':backend},cfg)
    audit=bundle.data_audit
    checks=run_checks(bundle,cfg);assert checks['status']=='ok',checks
    train_rows=[r for r in audit['records'] if r['split']=='train']
    audit['train_coverage_mean']=float(np.mean([r['target_coverage'] for r in train_rows]))
    assert audit['train_coverage_mean']>0.5,'Default protocol needs train-only review'
    audit['checks']=checks;audit['status']='data_validated'
    out=root/task;out.mkdir(exist_ok=True)
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    (out/'task_spec.json').write_text(json.dumps(audit['task_spec'],indent=2))
    lock['expand_1109']['raw_files'].update(audit['source_files'])
    print(task,'entities',audit['n_entities'],'mothers',audit['n_mothers'],'train coverage',audit['train_coverage_mean'],checks['status'],flush=True)
lock_path.write_text(json.dumps(lock,indent=2)+'\n')
(root/'download_record.json').write_text(json.dumps(lock['expand_1109'],indent=2))
