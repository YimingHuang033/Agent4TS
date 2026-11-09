"""Immutable v3 manifest and explicit attempts. No launch without audited data."""
import argparse,copy,hashlib,json,time,traceback,fcntl,tarfile,platform,importlib.metadata
from pathlib import Path
from agent4ts import config
from agent4ts.data import load_task
from agent4ts.action_space import ActionSpace
from agent4ts.schemas import PipelineSpec,DataView
from agent4ts.models.registry import runnable_ids
from agent4ts.baselines.greedy_search import run_matched
from agent4ts.baselines.tsci_adapted import run_tsci_adapted
from agent4ts.search import finalize,initial_spec
from agent4ts.cli import pipeline_from_dict
from agent4ts.degradation import apply_degradation,default_grid


def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def settings():
    cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3'])
    cfg['tasks'].pop('mimic_iv',None)
    for name in cfg['expansion']['new_tasks']:cfg['tasks'][name]={'backend':name.removesuffix('_forecast')}
    return cfg

def generate(cfg):
    audit=Path(cfg['expansion']['audit_root'])
    assert (audit/'debug_passed.json').exists(),'Complete CPU debug before freezing manifest'
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('src','scripts/gen_eval','config') for p in sorted(Path(folder).rglob('*')) if p.suffix in ('.py','.sh','.yaml')}
    code_hash=sha(sources);training_hash=sha({'resources':cfg['resources'],'defaults':cfg['evaluation']})
    bundles={t:load_task(s,cfg) for t,s in cfg['tasks'].items()}
    cells={}
    def add(phase,task,method,budget=12,f0='APN',degradation=None):
        if method=='tsci_adapted':method='tsci_model_only'  # same implementation/domain; reuse across phases
        base=bundles[task]
        for seed in cfg['seeds']:
            bundle=apply_degradation(base,degradation,seed=seed) if degradation else base
            bundle.spec.f0_model=f0
            space='model_only' if method.endswith('model_only') or method=='tsci_adapted' else 'data_only' if method.endswith('data_only') else 'full'
            identity={'dataset_version':bundle.spec.dataset_version,'task':task,'split_hash':sha(bundle.spec.split_ids),
                'protocol_version':cfg['expansion']['protocol_version'],'method':method,
                'action_space_hash':ActionSpace(space,initial_spec(bundle)).hash,'budget':budget,'f0':f0,'seed':seed,
                'degradation':degradation,'code_hash':code_hash,'model_training_hash':training_hash}
            key=sha(identity)[:24]
            if key not in cells:cells[key]={'id':key,'identity':identity,'space':space,'phases':[]}
            cells[key]['phases'].append(phase)
    for t in cfg['expansion']['new_tasks']:
        for m in ['fixed_'+x for x in runnable_ids()]+['random_full','agent_full','no_vision','no_semantics','one_shot','tsci_adapted']:add('new_data',t,m)
    for t in bundles:
        for m in ['agent_model_only','random_model_only','tsci_model_only','agent_full','random_full','greedy_full']:add('matched_space',t,m)
        for b in (4,8,12):
            for m in ['agent_full','random_full','greedy_full']:add('budget',t,m,b)
        for m in ['agent_model_only','agent_data_only','agent_full']:add('data_ablation',t,m)
        if t.startswith('physiome_'):
            for d in default_grid():
                for m in ['fixed_APN','fixed_DLinear','random_full','agent_full','tsci_adapted']:add('robustness',t,m,degradation=d)
    for t in ['physiome_dupont','epa_air_dallas','p12_robust','ushcn_forecast','pamap2_forecast']:
        for f0 in ['APN','DLinear','TiWeaver']:
            for m in ['agent_full','random_full','greedy_full']:add('initialization',t,m,f0=f0)
    return {'version':'manifest_1109_v1','source_hashes':sources,'code_hash':code_hash,'cells':list(cells.values())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['manifest','new_data','matched_space','budget','data_ablation','robustness','initialization'])
    parser.add_argument('--retry',help='JSON list of explicit failed cell IDs; all old attempts retained')
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    cfg=settings();root=Path(cfg['expansion']['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_file=root/'manifest.json'
        if not manifest_file.exists():
            manifest=generate(cfg)
            with tarfile.open(root/'source_snapshot.tar.gz','w:gz') as archive:
                for file in manifest['source_hashes']:archive.add(file,arcname=file)
            env={'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ('numpy','pandas','torch','openai')}}
            (root/'environment.json').write_text(json.dumps(env,indent=2))
            manifest_file.write_text(json.dumps(manifest,indent=2))
            (root/'config.json').write_text(json.dumps({k:v for k,v in cfg.items() if k!='llm'},indent=2))
            (root/'llm.json').write_text(json.dumps({k:v for k,v in config.llm_settings(cfg).items() if 'key' not in k},indent=2))
        manifest=json.loads(manifest_file.read_text())
        for file,digest in manifest['source_hashes'].items():
            assert Path(file).exists() and hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,f'Code/config drift: {file}. Use a new version/root; do not overwrite manifest.'
        if args.phase=='manifest':print('manifest cells:',len(manifest['cells']));return
        retry=set(json.loads(Path(args.retry).read_text())) if args.retry else set();bundles={};ran=0
        for cell in manifest['cells']:
            if args.phase not in cell['phases']:continue
            ident=cell['identity'];directory=root/'cells'/cell['id'];attempts=sorted(directory.glob('attempt*/result.json'))
            if attempts:
                last=json.loads(attempts[-1].read_text())
                if last['status']=='ok' or cell['id'] not in retry:continue
            elif args.retry and cell['id'] not in retry:continue
            number=len(list(directory.glob('attempt*')))+1;out=directory/f'attempt{number:03}';out.mkdir(parents=True,exist_ok=False)
            (out/'identity.json').write_text(json.dumps(ident,indent=2));(out/'config.json').write_text((root/'config.json').read_text())
            (out/'source_hashes.json').write_text(json.dumps(manifest['source_hashes'],indent=2))
            started=time.monotonic();row={'id':cell['id'],**ident,'attempt':number,'status':'running'}
            (out/'result.json').write_text(json.dumps(row,indent=2))
            try:
                task=ident['task']
                if task not in bundles:bundles[task]=load_task(cfg['tasks'][task],cfg)
                bundle=copy.deepcopy(bundles[task]);bundle.spec.f0_model=ident['f0']
                if ident['degradation']:bundle=apply_degradation(bundle,ident['degradation'],seed=ident['seed'])
                local=copy.deepcopy(cfg);local['budget']['trials']=ident['budget'];method=ident['method'];seed=ident['seed'];search_started=time.monotonic()
                if method.startswith('fixed_'):candidates=[PipelineSpec('fixed',DataView(),method[6:],{},seed)]
                else:
                    if method.startswith('tsci'):result=run_tsci_adapted(bundle,local,out,seed=seed)
                    else:
                        policy='random' if method.startswith('random') else 'greedy' if method.startswith('greedy') else 'agent'
                        result=run_matched(bundle,local,out,seed,policy,cell['space'],method if method in ('no_vision','no_semantics','one_shot') else None)
                    (out/'search.json').write_text(json.dumps(result,indent=2,default=str))
                    if result.get('action_space_hash')!=ident['action_space_hash']:raise RuntimeError('Action space hash mismatch')
                    if 'unparseable' in result.get('stop_reason',''):raise RuntimeError(result['stop_reason'])
                    row['used']=result['used'];row['llm_wall_s']=result.get('llm_wall_s')
                    ranked=sorted((x for x in result['archive'] if x['status']=='ok' and x['metrics'].get('normalized_mae') is not None),key=lambda x:(x['metrics']['normalized_mae'],x['fingerprint']))
                    candidates=[pipeline_from_dict(x['pipeline']) for x in ranked[:3]]
                row['search_wall_s']=time.monotonic()-search_started
                finalize_started=time.monotonic();final=finalize(bundle,local,out,candidates,seed=seed)
                row.update(status=final['status'],test=final.get('test'),finalize_wall_s=time.monotonic()-finalize_started)
            except Exception as exc:
                row.update(status='error',test=None,error=f'{type(exc).__name__}: {exc}')
                (out/'error.txt').write_text(traceback.format_exc())
            row['wall_s']=time.monotonic()-started
            (out/'result.json').write_text(json.dumps(row,indent=2,default=str))
            print(cell['id'],ident['task'],ident['method'],row['status'],flush=True)
            ran+=1
            if args.limit and ran>=args.limit:break

if __name__=='__main__':main()
