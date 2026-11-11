"""MetaCaster-adapted: released MGAgent + trained harness; shared ATS forecasters.

No HPAgent retraining on evaluation tasks. The official chat-completions loop
is reused with an Ark-only client; its historical filename does not start vLLM.
Generated code runs in a networkless, credentialless bwrap mount namespace.
Only train support is mounted. FTAgent's pool/selection is adapted to the ATS
model pool and normalized MAE; this is not the paper's LT-Lib reproduction.
"""
import copy, hashlib, json, os, re, subprocess, sys, types
from pathlib import Path
import numpy as np
from ..config import llm_settings
from ..harness import Harness
from ..schemas import DataView, PipelineSpec
from ..action_space import ActionSpace
from ..search import initial_spec, finalize


def support_tensor(windows, norm, n_history, n_future, duration, horizon):
    """Interpolate history and training labels SEPARATELY, in original units.

    This dense view is generation-only. Reconstructed training examples retain
    template observation masks and label masks; missing labels stay unscored.
    """
    hg=np.linspace(0,duration,n_history);fg=np.linspace(0,horizon,n_future)
    tensors=[]
    for lw in windows:
        w=lw.window;cols=[]
        for row,vid in enumerate(w.variable_ids):
            vid=int(vid);m=w.mask[row].astype(bool)&np.isfinite(w.x[row])
            tx=w.times[m];vx=norm.inverse(vid,w.x[row,m]);order=np.argsort(tx)
            hist=np.interp(hg,tx[order],vx[order]) if len(tx) else np.full(n_history,norm.mean.get(vid,0.))
            m=(lw.query_var_ids==vid)&lw.y_scored&np.isfinite(lw.y_true)
            tx=lw.query_times[m];vx=lw.y_true[m];order=np.argsort(tx)
            fut=np.interp(fg,tx[order],vx[order]) if len(tx) else np.full(n_future,hist[-1])
            cols.append(np.r_[hist,fut])
        tensors.append(np.stack(cols,axis=1))
    return np.asarray(tensors,dtype=np.float32),hg,fg


def synthetic_windows(data, templates, norm, hg, fg):
    out=[];nh=len(hg)
    for i,arr in enumerate(data):
        lw=copy.deepcopy(templates[i%len(templates)]);w=lw.window
        w.entity_id=-(i+1)
        for row,vid in enumerate(w.variable_ids):
            w.x[row]=norm.transform(int(vid),np.interp(w.times,hg,arr[:nh,row]))
            mask=lw.query_var_ids==vid
            lw.y_true[mask]=np.interp(lw.query_times[mask],fg,arr[nh:,row])
        lw.y_true[~lw.y_scored]=np.nan
        out.append(lw)
    return out


def sandbox_python(code, workspace):
    """No home, project, data store, network, GPU or secrets visible to code."""
    workspace=Path(workspace).resolve()
    script=workspace/'_execution.py';script.write_text(code)
    prefix=Path(sys.prefix)
    cmd=['bwrap','--unshare-all','--die-with-parent','--new-session',
         '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
         '--ro-bind',str(prefix),str(prefix),'--bind',str(workspace),'/work',
         '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/work',
         '--clearenv','--setenv','PATH',str(prefix/'bin')+':/usr/bin',
         '--setenv','HOME','/tmp','--setenv','CUDA_VISIBLE_DEVICES','',
         '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
         '--',sys.executable,'/work/_execution.py']
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
        return (f'exit_code={p.returncode}\n'+p.stdout+'\n'+p.stderr)[-16000:]
    except subprocess.TimeoutExpired:return 'Error: 120 second timeout'


def generate(samples, cfg, workspace, seed):
    mc=cfg['metacaster'];source=Path(mc['source']);workspace=Path(workspace).resolve()
    rev=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if rev!=mc['revision']:raise RuntimeError('MetaCaster source revision mismatch')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise RuntimeError('MetaCaster vendor must be unmodified')
    sys.path.insert(0,str(source))
    from generation.core import loop_vllm as upstream
    from generation.tools import build_tools
    from generation.tools.skill import SkillLoader
    from generation.agent import _validate_generated_artifact
    from openai import OpenAI
    llm=llm_settings(cfg)
    if not llm['base_url'].startswith('https://ark.cn-beijing.volces.com/'):
        raise ValueError('This baseline requires the remote Ark endpoint')
    client=OpenAI(api_key=llm['api_key'],base_url=llm['base_url'],max_retries=0,timeout=180)
    workspace.mkdir(parents=True,exist_ok=False);inp=workspace/'input';inp.mkdir()
    np.save(inp/'few_shot.npy',samples)
    meta={'seq_len':mc['history_bins'],'pred_len':mc['future_bins'],'C_eff':samples.shape[2],
          'n_target':mc['n_target'],'seed':seed,'domain':cfg.get('_task_id','irregular forecasting'),
          'preprocessing':'Train support only; history/future interpolated separately; each channel divided by train standard deviation, without centering (zeros preserved).' }
    (inp/'meta.json').write_text(json.dumps(meta))
    (inp/'context.txt').write_text(json.dumps(cfg.get('_context',{}),indent=2))
    skill=SkillLoader(source/'harness/skills');tools,handlers=build_tools(skill,workspace=workspace)
    handlers['run_python']=lambda code:sandbox_python(code,workspace)
    system=(source/'harness/core/router.md').read_text().format(input_dir='input',output_dir='{output_dir}',skill_descriptions=skill.get_descriptions())
    calls=[]
    def create(**kwargs):
        if len(calls)>=mc['max_calls']:raise RuntimeError('LLM API call budget exhausted')
        record={'call':len(calls)+1};calls.append(record)
        try:
            response=client.chat.completions.create(**kwargs,max_tokens=mc['max_tokens'],extra_body={'thinking':{'type':'disabled'}})
            record['usage']=response.usage.model_dump() if response.usage else None
            return response
        except Exception as exc:
            record['error_type']=type(exc).__name__;raise
        finally:(workspace/'api_usage.json').write_text(json.dumps(calls,indent=2))
    proxy=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    previous=upstream._build_client;upstream._build_client=lambda:proxy
    # Execute the released validation code independently of the agent's report.
    skill_text=(source/'harness/skills/regime_aware_bootstrap_suite/SKILL.md').read_text()
    released_code=re.search(r'```python\n(.*?)```',skill_text,re.S).group(1)
    (workspace/'released_skill.py').write_text(released_code)
    ns={};exec(compile(released_code,'official_metacaster_skill','exec'),ns)
    try:
        upstream.agent_loop(f'Generate exactly {mc["n_target"]} task-adaptive training windows. Input: input/few_shot.npy, input/meta.json, and input/context.txt. Use random seed {seed}. The exact released Python functions are already available in released_skill.py; import them instead of transcribing the skill. Run the released skill, validate all gates, save dataset.npy and validation_report.json. The execution workspace is /work; use relative paths.',system=system,model=llm['model'],max_turns=mc['max_calls'],tools=tools,tool_handlers=handlers,output_dir=str(workspace))
    finally:upstream._build_client=previous
    artifact=_validate_generated_artifact(workspace,inp/'meta.json',mc['n_target'])
    data=np.load(artifact,allow_pickle=False);passed,report=ns['validate_generated'](samples,data)
    (workspace/'independent_validation.json').write_text(json.dumps({'passed':passed,**report},indent=2))
    if not passed:raise ValueError('Official independent generation validation failed: '+str(report['fail']))
    return data,calls


def run_metacaster(bundle,cfg,run_dir,seed=0,control=False):
    run_dir=Path(run_dir);mc=cfg['metacaster'];h=Harness(bundle,cfg,run_dir,seed=seed)
    view=DataView();train=h.windows('train',view)
    rng=np.random.default_rng(seed);indices=rng.permutation(len(train))[:min(mc['support_size'] or len(train),len(train))]
    support=[train[int(i)] for i in indices]
    samples,hg,fg=support_tensor(support,h.norm,mc['history_bins'],mc['future_bins'],bundle.spec.history_duration,bundle.spec.horizon)
    scales=np.asarray([h.norm.std.get(int(v),1.) for v in support[0].window.variable_ids],dtype=np.float32)
    samples=samples/scales[None,None,:]
    (run_dir/'support.json').write_text(json.dumps({'entity_ids':[w.window.entity_id for w in support], 'tensor_sha256':hashlib.sha256(samples.tobytes()).hexdigest(),'count':len(support),'seed':seed,'channel_scales':scales.tolist()},indent=2))
    calls=[]
    if control:training=support
    else:
        data,calls=generate(samples,{**cfg,'_task_id':bundle.spec.task_id,'_context':{'variables':bundle.card.variables,'time_semantics':bundle.card.time_semantics,'protocol_notes':bundle.spec.notes}},run_dir/'mgagent',seed)
        training=synthetic_windows(data*scales[None,None,:],support,h.norm,hg,fg)
    key=('train',json.dumps(view.to_dict(),sort_keys=True));h._windows[key]=training
    h.used['llm_calls']=len(calls)
    domain=ActionSpace('model_only',initial_spec(bundle))
    for i,mid in enumerate(mc['models']):
        if h.remaining()['trials']<=0 or h.remaining()['gpu_hours']<=0:break
        spec=domain.canonical(PipelineSpec(f'metacaster_{i}',view,mid,{},seed))
        fb=h.run_trial(spec);h.update_incumbent(spec,fb)
    valid=sorted([x for x in h.archive if x['status']=='ok'],key=lambda x:x['metrics']['normalized_mae'])
    if not valid:raise RuntimeError('No successfully trained forecaster')
    from ..cli import pipeline_from_dict
    candidates=[pipeline_from_dict(x['pipeline']) for x in valid[:3]]
    # Must reuse the fitted synthetic/support models; finalizer must never refit on full train.
    for spec in candidates:
        fp=spec.fingerprint(h._upstream_hash(),bundle.spec.dataset_version,h._splits_hash())
        if not (run_dir/'_models'/f'{fp}.pkl').exists():raise RuntimeError('Missing fitted artifact')
    result=finalize(bundle,cfg,run_dir,candidates,seed=seed)
    return {'label':'Real-support-pool' if control else 'MetaCaster-adapted',
            'status':result['status'],'archive':h.archive,'used':h.used,'finalize':result,
            'protocol':'metacaster_ats_v1', 'support_size_requested':mc['support_size'],'upstream_revision':mc['revision'],
            'adaptations':['dense train-only generator view','shared ATS forecaster pool','normalized MAE selection','confirm top3 then test once','Ark API backbone'],
            'model_pool':mc['models'],'support_count':len(support)}
