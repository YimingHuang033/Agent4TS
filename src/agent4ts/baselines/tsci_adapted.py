"""TSci-adapted: upstream prompts and validation orchestration, unified executor.

Exact upstream validation run/test/select methods are loaded from the pinned
checkout. Replaced boundaries: OpenAI-compatible transport, data observation,
hyperparameter execution/splits, and fallback (errors, never synthetic scores).
This is not the unmodified TSci framework or its original benchmark protocol.
"""
import ast, __future__, json, logging, itertools, types
from pathlib import Path
from ..controller import Controller
from ..harness import Harness
from ..schemas import PipelineSpec,DataView
from ..models.registry import runnable_ids,CATALOG
from ..search import initial_spec
from .. import observe

def _upstream(root):
    namespace={'logger':logging.getLogger('TSci-adapted')}
    for name, constants, functions in [
        ('analysis_agent',['ANALYSIS_SYSTEM_PROMPT'],['get_analysis_prompt']),
        ('validation_agent',['VALIDATION_SYSTEM_PROMPT'],['get_model_selection_prompt'])]:
        path=root/'agents'/f'{name}.py';tree=ast.parse(path.read_text())
        nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):nodes.append(node)
            if isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
            if isinstance(node,ast.ClassDef) and node.name=='ValidationAgent':
                node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('run','_test_models_on_validation_data','_select_best_models_from_testing')]
                nodes.append(node)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace

def run_tsci_adapted(bundle,cfg,run_dir,seed=0):
    run_dir=Path(run_dir);h=Harness(bundle,cfg,run_dir,seed=seed)
    ctrl=Controller(cfg,run_dir);ctrl.call_budget=h
    upstream=_upstream(Path(cfg['resources']['tsci_dir'])/'time_series_agent')
    f0=initial_spec(bundle);f0.seed=seed;h.f0_id=f0.pipeline_id
    from ..action_space import ActionSpace
    domain=ActionSpace("model_only",f0); f0=domain.canonical(f0);seen={domain.key(f0)}
    h.write_trace({"event":"space","action_space_hash":domain.hash,"descriptor":domain.descriptor(),"f0":f0.to_dict()})
    fb=h.run_trial(f0);h.update_incumbent(f0,fb)
    stats=observe.statistics(bundle,DataView(),'search')
    card=observe.semantic_card(bundle)
    figs=observe.make_plots(bundle,DataView(),'search',run_dir/'figures',seed=seed)
    import base64
    content=[{'type':'text','text':upstream['get_analysis_prompt'](json.dumps({'statistics':stats,'semantics':card}),figs)}]
    for fig in figs:
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(fig['path']).read_bytes()).decode()}})
    analysis=ctrl.request([{'role':'system','content':upstream['ANALYSIS_SYSTEM_PROMPT']},{'role':'user','content':content}])
    if not analysis:raise RuntimeError('TSci analysis API failure')
    agent=upstream['ValidationAgent'].__new__(upstream['ValidationAgent']);agent.config={'k_models':3};agent.n_candidates=3;agent.k_models=3
    def parse(self,analysis_result,available_models):
        prompt=upstream['get_model_selection_prompt'](analysis_result,available_models,3)
        prompt+='\nOnly these hyperparameter names and finite values are allowed: '+json.dumps({m:CATALOG[m]['options'] for m in available_models})
        result=ctrl.request([{'role':'system','content':upstream['VALIDATION_SYSTEM_PROMPT']},{'role':'user','content':prompt}],lambda raw:json.loads(raw[raw.index('{'):raw.rindex('}')+1]))
        h.write_trace({'event':'tsci_selection','selection':result})
        return result
    def optimize(self,data,mid,space):
        if mid not in runnable_ids():raise ValueError('Unregistered TSci model')
        options=CATALOG[mid]['options'];keys=list(space)
        for key,values in space.items():
            if key not in options:raise ValueError('Unsupported TSci model option '+key)
            if not isinstance(values,list):space[key]=[values]
            if any(v not in options[key] for v in space[key]):raise ValueError('Unsupported TSci option value')
        best=None
        for vals in itertools.product(*(space[k] for k in keys)):
            if h.exhausted():break
            spec=PipelineSpec(f'tsci_{len(h.archive)}',DataView(),mid,dict(zip(keys,vals)),seed)
            spec=domain.canonical(spec)
            key=domain.key(spec)
            if key in seen:
                fb=h.prior_trial(spec)
                h.write_trace({'event':'duplicate','fingerprint':key,'policy':'tsci'})
            else:
                seen.add(key); fb=h.run_trial(spec);h.update_incumbent(spec,fb)
            if fb is None:continue
            if fb.status=='ok' and (best is None or fb.metrics['normalized_mae']<best[1]['mae']):
                import numpy as np
                ape=[]
                for lw,pred in zip(h.windows('search',spec.data_view),fb._artifacts['preds']):
                    truth=lw.y_true; mask=lw.y_scored & np.isfinite(truth) & (np.abs(truth)>1e-8)
                    ape.extend((np.abs((np.asarray(pred)[mask]-truth[mask])/truth[mask])*100).tolist())
                best=(spec.model_config,{'mae':fb.metrics['normalized_mae'],'mse':fb.metrics['normalized_mse'],'mape':float(np.mean(ape)) if ape else float('nan')})
        if best is None:raise RuntimeError('No successful TSci candidate')
        return best
    def fail(self,*args):raise RuntimeError('TSci upstream fallback refused: no synthetic validation scores')
    agent._parse_structured_response=types.MethodType(parse,agent)
    agent._optimize_model_hyperparameters=types.MethodType(optimize,agent)
    agent._generate_fallback_models=types.MethodType(fail,agent)
    selected=agent.run(analysis,runnable_ids(),None)
    if not selected:raise RuntimeError('TSci returned no candidates')
    valid=[a for a in h.archive if a['status']=='ok' and a['metrics'].get('normalized_mae') is not None]
    if not any(a['pipeline']['pipeline_id']!='p0_f0' for a in valid):
        raise RuntimeError('No successful LLM-selected TSci candidate beyond f0')
    winner=min(valid,key=lambda a:a['metrics']['normalized_mae'])
    return {'task_id':bundle.spec.task_id,'seed':seed,'label':'TSci-adapted','incumbent':winner['pipeline'],'archive':h.archive,'used':h.used,'f0_metric':h.f0_metric(),'incumbent_metric':winner['metrics']['normalized_mae'],'upstream_selected':selected,'action_space_hash':domain.hash}
