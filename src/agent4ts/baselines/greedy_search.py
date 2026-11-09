"""Shared v3 loop for agent, random and k-neighbor greedy search."""
import random,time
from pathlib import Path
from .. import observe
from ..action_space import ActionSpace
from ..controller import Controller
from ..harness import Harness
from ..schemas import Observation,Feedback
from ..search import initial_spec,apply_action


def run_matched(bundle,cfg,run_dir,seed=0,policy='greedy',mode='full',ablation=None):
    h=Harness(bundle,cfg,Path(run_dir),seed=seed);f0=initial_spec(bundle);f0.seed=seed
    space=ActionSpace(mode,f0);f0=space.canonical(f0);inc=f0;h.f0_id=f0.pipeline_id
    ctrl=Controller(cfg,Path(run_dir));ctrl.call_budget=h;rng=random.Random(seed)
    fb=h.run_trial(f0);h.update_incumbent(f0,fb);seen={space.key(f0)}
    h.write_trace({'event':'space','action_space_hash':space.hash,'descriptor':space.descriptor(),'f0':f0.to_dict()})
    decisions=0;reason='budget_exhausted';llm_time=0.
    while not h.exhausted():
        decisions+=1
        if decisions>cfg['budget']['llm_calls']*4 and policy=='agent':reason='decision_cap';break
        neighbors=space.neighbors(inc,h,seen)
        if not neighbors:reason='no_legal_unseen_neighbor';break
        parent=inc
        if policy=='agent':
            figs=[] if ablation=='no_vision' else observe.make_plots(bundle,inc.data_view,'search',Path(run_dir)/'figures',seed=seed)
            obs=Observation(task_id=bundle.spec.task_id,round_index=decisions-1,
                dataset_card={} if ablation=='no_semantics' else observe.semantic_card(bundle),
                statistics=observe.statistics(bundle,inc.data_view,'search'),plot_paths=figs,
                archive_summary=[{"pipeline": e["pipeline"], "status": e["status"], "normalized_mae": e["metrics"].get("normalized_mae")} for e in h.archive],incumbent=inc.to_dict(),last_feedback=h.last_feedback.to_dict() if h.last_feedback else None,
                budget=h.remaining(),allowed_ops=['STOP','INSPECT']+(['SELECT_MODEL','SET_MODEL_OPTION'] if mode!='data_only' else [])+(['SET_TRANSFORM','SELECT_DATA'] if mode!='model_only' else []),
                action_space=space.descriptor())
            t=time.monotonic();action=ctrl.propose(obs);llm_time+=time.monotonic()-t
            h.write_trace({'event':'propose','decision':decisions,'action':action.to_dict(),'figures':[f.get('sha256') for f in figs],
                'semantics_present':bool(obs.dataset_card),'action_space_hash':space.hash})
            if action.op=='STOP':reason=action.expected_effect or 'active_stop';break
            if action.op=='INSPECT':
                if ablation=='one_shot':reason='one_shot';break
                continue
            checks=h.validate(action,parent)
            if not space.permits(action) or not h.checks_ok(checks):
                h.last_feedback=Feedback(status='rejected',checks=checks,error='Outside shared space or incompatible',remaining_budget=h.remaining())
                h.write_trace({'event':'rejected','decision':decisions,'checks':checks})
                if ablation=='one_shot':reason='one_shot';break
                continue
            candidate=space.canonical(apply_action(action,parent,len(h.archive)))
            key=space.key(candidate)
            if key in seen:
                h.last_feedback=Feedback(status='rejected',error='Already tried canonical candidate',remaining_budget=h.remaining())
                h.write_trace({'event':'duplicate','decision':decisions,'fingerprint':key})
                if ablation=='one_shot':reason='one_shot';break
                continue
            chosen=[(key,action,candidate)]
        elif policy in ('random','greedy'):
            rng.shuffle(neighbors)
            chosen=neighbors[:1 if policy=='random' else cfg['expansion']['greedy_k']]
            h.write_trace({'event':'neighbors','decision':decisions,'parent':space.key(parent),'order':[x[0] for x in neighbors], 'selected':[x[0] for x in chosen]})
        else:raise ValueError(policy)
        evaluated=[]
        for key,action,candidate in chosen:
            if h.exhausted():break
            candidate.pipeline_id=f'p{len(h.archive)}_{candidate.model_id}'
            candidate.parent_id=parent.pipeline_id
            seen.add(key);feedback=h.run_trial(candidate)
            metric=feedback.metrics.get('normalized_mae')
            evaluated.append((float('inf') if metric is None else metric,key,candidate,feedback))
            h.write_trace({'event':'attempt','trial_index':h.used['trials'],'decision':decisions,
                'action':action.to_dict(),'parent':space.key(parent),'fingerprint':key,'status':feedback.status,
                'metrics':feedback.metrics,'error':feedback.error,'used':dict(h.used),'wall_s':feedback.cost.get('wall_s')})
        if evaluated:
            _,_,best,feedback=min(evaluated,key=lambda x:(x[0],x[1]))
            accepted,_=h.update_incumbent(best,feedback)
            if accepted:inc=best
            h.write_trace({'event':'selection','decision':decisions,'accepted':accepted,'incumbent':space.key(inc),'best_so_far':h._incumbent_metric()})
        if ablation=='one_shot':reason='one_shot';break
    h.write_trace({'event':'stop','reason':reason})
    return {'archive':h.archive,'incumbent':inc.to_dict(),'f0_metric':h.f0_metric(),'used':h.used,
            'incumbent_metric':h._incumbent_metric(),'action_space_hash':space.hash,'decisions':decisions,'stop_reason':reason,'llm_wall_s':llm_time}


def run_greedy_search(bundle,cfg,run_dir,seed=0):
    return run_matched(bundle,cfg,run_dir,seed,policy='greedy')
