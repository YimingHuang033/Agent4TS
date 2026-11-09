"""Resumable v2 experiment matrix; every cell retains status, seed and provenance."""
import copy, json, time, traceback
from pathlib import Path
import numpy as np
from agent4ts import config as cfgmod
from agent4ts.data import load_task
from agent4ts.harness import Harness
from agent4ts.schemas import PipelineSpec, DataView
from agent4ts.search import run_search, finalize
from agent4ts.cli import pipeline_from_dict
from agent4ts.baselines.random_search import run_random_search
from agent4ts.degradation import apply_degradation, default_grid
from agent4ts.tests_checks import run_checks
from agent4ts.models.registry import runnable_ids
from agent4ts.baselines.tsci_adapted import run_tsci_adapted

cfg = cfgmod.load_config('gen_eval/main')
root = Path(cfg['paths']['results_root']) / 'gen_eval' / 'v2_ark_20261107'
root.mkdir(parents=True, exist_ok=True)
import hashlib
(root/'source_hashes.json').write_text(json.dumps({str(path.relative_to(Path(__file__).parents[2])):hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parents[2].joinpath('src').rglob('*.py')},indent=2))
(root/'config.json').write_text(json.dumps({k:v for k,v in cfg.items() if k != 'llm'},indent=2,default=str))
(root/'llm.json').write_text(json.dumps({k:v for k,v in cfg['llm'].items() if 'key' not in k},indent=2))
rows=[]

def run_cell(bundle, task, seed, method, degradation='none'):
    out=root/task/degradation/method/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    record=out/'result.json'
    if record.exists():
        rows.append(json.loads(record.read_text()));return
    started=time.time()
    row={'task':task,'seed':seed,'method':method,'degradation':degradation,'metric_version':cfg['evaluation']['metric_version']}
    try:
        if method.startswith('fixed_'):
            spec=PipelineSpec('fixed',DataView(),method[6:],{},seed)
            result=finalize(bundle,cfg,out,[spec],seed=seed)
        else:
            search=(run_tsci_adapted(bundle,cfg,out,seed=seed) if method=='tsci_adapted' else
                    run_random_search(bundle,cfg,out,seed=seed) if method=='random' else
                    run_search(bundle,cfg,out,seed=seed,ablation=None if method=='agent' else method))
            if method != 'random':
                trace = [json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
                if not any(e.get('event') == 'llm_usage' for e in trace):
                    raise RuntimeError('No successful LLM response; not a valid agent experiment')
                if any(e.get('event') == 'stop' and 'unparseable' in e.get('reason','') for e in trace):
                    raise RuntimeError('LLM action parse failure; do not count fallback as successful agent run')
            (out/'search.json').write_text(json.dumps(search,indent=2,default=str))
            candidates=sorted((x for x in search['archive'] if x['status']=='ok' and x['metrics'].get('normalized_mae') is not None),key=lambda x:x['metrics']['normalized_mae'])[:3]
            result=finalize(bundle,cfg,out,[pipeline_from_dict(x['pipeline']) for x in candidates],seed=seed)
            row['search_used']=search['used']; row['search_f0']=search['f0_metric']
        row.update(status=result['status'],test=result.get('test'))
    except Exception as exc:
        row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        (out/'error.txt').write_text(traceback.format_exc())
    row['wall_s']=time.time()-started
    record.write_text(json.dumps(row,indent=2,default=str));rows.append(row)
    print(json.dumps({k:row[k] for k in ['task','seed','method','degradation','status','wall_s']}),flush=True)
    (root/'progress.json').write_text(json.dumps(rows,indent=2,default=str))

# Complete each stage across all available tasks before expensive extensions.
bundles={}
for task, spec in sorted(cfg['tasks'].items(), key=lambda item: ({'physiome':0,'timeimm':1,'p12':2}.get(item[1]['backend'],3),item[0])):
    try:
        bundle=load_task(spec,cfg); bundle.spec.f0_model='APN'; checks=run_checks(bundle,cfg)
        if checks['status']!='ok': raise ValueError(str(checks))
        bundles[task]=bundle
    except Exception as exc:
        (root/f'{task}_status.json').write_text(json.dumps({'status':'blocked','reason':str(exc)},indent=2))
for task,bundle in bundles.items():
    for seed in cfg['seeds']:
        for method in ['fixed_'+mid for mid in runnable_ids()]+['random','agent','no_vision','no_semantics','one_shot','tsci_adapted']:
            run_cell(bundle,task,seed,method)
for task,bundle in bundles.items():
    if bundle.spec.backend!='physiome': continue
    for seed in cfg['seeds']:
        for degradation in default_grid():
            degraded=apply_degradation(bundle,degradation,seed=seed)
            for method in ['fixed_APN','fixed_DLinear','random','agent']:
                run_cell(degraded,task,seed,method,degradation['tag'])
summary=[]
for task in bundles:
    for method in sorted(set(r['method'] for r in rows)):
        for degradation in sorted(set(r['degradation'] for r in rows)):
            cells=[r for r in rows if r['task']==task and r['method']==method and r['degradation']==degradation]
            if not cells: continue
            vals=[r['test']['metrics']['normalized_mae'] for r in cells if (r.get('test') or {}).get('status')=='ok' and r['test']['metrics']['normalized_mae'] is not None]
            summary.append({'task':task,'method':method,'degradation':degradation,'n_ok':len(vals),'n_total':len(cells),'mean':float(np.mean(vals)) if vals else None,'std':float(np.std(vals,ddof=1)) if len(vals)>1 else None})
(root/'summary.json').write_text(json.dumps(summary,indent=2))
(root/'DONE').write_text('Experiment matrix finished; inspect blocked and error statuses before interpreting results.\n')
