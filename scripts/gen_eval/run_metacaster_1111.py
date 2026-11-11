"""Explicit, resumable paired baseline queue; no retries overwrite old attempts."""
import argparse, copy, fcntl, hashlib, json, time, traceback
from pathlib import Path
from agent4ts.config import load_config
from agent4ts.data import load_task
from agent4ts.baselines.metacaster import run_metacaster

def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--shots',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--plan',action='store_true');p.add_argument('--task');p.add_argument('--seed',type=int);p.add_argument('--limit',type=int);p.add_argument('--retry',action='store_true');a=p.parse_args()
    cfg=load_config('gen_eval/main',extra=['gen_eval/expand_v3','gen_eval/metacaster_1111'])
    cfg['tasks'].pop('mimic_iv',None)
    if a.shots is not None:
        if a.shots<2:raise ValueError('shots must be at least 2')
        cfg['metacaster']['support_size']=a.shots
        cfg['metacaster']['root']+=f'_k{a.shots}'
    if a.smoke:
        import os
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Smoke must hide GPUs')
        cfg['metacaster'].update(root='results/smoke/metacaster_1111',models=['DLinear','MeanValue'],support_size=30,n_target=64,max_calls=24)
    mc=cfg['metacaster'];root=Path(mc['root']);root.mkdir(parents=True,exist_ok=True)
    tasks=[a.task] if a.task else (['physiome_dupont'] if a.smoke else mc['tasks'])
    seeds=[a.seed] if a.seed is not None else ([0] if a.smoke else cfg['seeds'])
    for task in tasks:
        if task not in mc['tasks'] or task not in cfg['tasks']:raise ValueError('Unknown task: '+task)
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('src','config','scripts/gen_eval') for f in sorted(Path(folder).rglob('*')) if f.suffix in ('.py','.sh','.yaml')}
    protocol={'cfg':{k:v for k,v in cfg.items() if k!='llm'},'llm':{k:v for k,v in cfg['llm'].items() if 'key' not in k},'source_hashes':hashes}
    identity=sha(protocol)
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=root/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['identity']!=identity:raise RuntimeError('Frozen MetaCaster code/config changed; use a new root')
        else:manifest.write_text(json.dumps({'identity':identity,**protocol},indent=2))
        planned=[(t,s,m) for t in tasks for s in seeds for m in ('real_support_pool','metacaster')]
        if a.plan:print(json.dumps({'root':str(root),'cells':planned,'count':len(planned)},indent=2));return
        done=0;failed=0
        for task,seed,method in planned:
            cell=root/task/f'seed{seed}'/method;old=sorted(cell.glob('attempt*/result.json'))
            if any(json.loads(x.read_text()).get('status')=='ok' for x in old):continue
            attempts=sorted(cell.glob('attempt*'))
            if attempts and not a.retry:continue
            run=cell/f'attempt{len(attempts)+1:03d}';run.mkdir(parents=True)
            result={'status':'running','task':task,'seed':seed,'method':method,'identity':identity,'started':time.time()}
            (run/'result.json').write_text(json.dumps(result,indent=2))
            try:
                bundle=load_task(cfg['tasks'][task],cfg);bundle.spec.task_id=task
                result.update(run_metacaster(bundle,copy.deepcopy(cfg),run,seed,control=method=='real_support_pool'))
            except Exception as exc:
                result.update(status='error',error=f'{type(exc).__name__}: {exc}')
                (run/'error.txt').write_text(traceback.format_exc())
            failed+=result['status']!='ok'
            result['finished']=time.time();(run/'result.json').write_text(json.dumps(result,indent=2,default=str))
            print(task,seed,method,result['status'],flush=True);done+=1
            if a.limit and done>=a.limit:break
        if failed:raise SystemExit(1)
if __name__=='__main__':main()
