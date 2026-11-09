"""Audited raw-source custom forecasts. No interpolation or global scaling.

Fixed nonoverlapping windows retain raw sample times and masks. Mother station /
subject split is assigned before windowing, identical for all experiment seeds.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib, json
import numpy as np
from ..schemas import DatasetCard, EventTable, QueryPolicy, TaskBundle, TaskSpec, Track


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def build(records, variables, spec, cfg, source_files):
    """records: (mother, absolute start, relative times, values, duration)."""
    opts=cfg['expansion']; fraction=opts['history_fraction']; buckets=opts['query_buckets']
    mothers=sorted({r[0] for r in records})
    order=np.random.default_rng(opts['split_seed']).permutation(mothers).tolist()
    a,b,c=[int(len(order)*x) for x in (.6,.8,.9)]
    groups=dict(zip(('train','search','confirm','test'),[order[:a],order[a:b],order[b:c],order[c:]]))
    assert all(groups.values()),'Need enough mother entities for four independent splits'
    mother_split={m:s for s,ms in groups.items() for m in ms}
    splits={s:[] for s in groups}; histories=[]; targets=[]; anchors={}; audit_rows=[]
    durations={float(r[4]) for r in records}; assert len(durations)==1
    duration=durations.pop(); cutoff=duration*fraction; horizon=duration-cutoff
    edges=np.linspace(cutoff,duration,buckets+1)
    def event(eid,t,v,y):
        return EventTable(np.full(len(y),eid,dtype=np.int64),np.asarray(t,float),np.asarray(t,float),np.asarray(v,np.int64),np.asarray(y,float))
    for eid,(mother,start,t,x,dur) in enumerate(records):
        assert np.isfinite(t).all() and np.all(np.diff(t)>0), 'Nonmonotonic/duplicate source timestamps'
        assert t.min()>=0 and t.max()<=dur
        hist=(t<=cutoff); finite=np.isfinite(x)
        ri,vi=np.where(finite & hist[:,None]); histories.append(event(eid,t[ri],vi,x[ri,vi]))
        qt=[];qv=[];qy=[]
        for bi,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            for v,meta in enumerate(variables.values()):
                take=(t>lo)&(t<=hi)&finite[:,v]
                if not take.any():continue
                yy=x[take,v]; agg=meta.get('aggregation','mean')
                # Interval totals/extrema only scored with complete daily coverage.
                if meta.get('require_complete_daily') and take.sum()!=round(hi-lo):continue
                value={'mean':np.mean,'sum':np.sum,'max':np.max,'min':np.min,'last':lambda a:a[-1]}[agg](yy)
                qt.append(hi);qv.append(v);qy.append(value)
        targets.append(event(eid,qt,qv,qy)); anchors[eid]=cutoff
        splits[mother_split[mother]].append(eid)
        audit_rows.append({'entity_id':eid,'mother_id':mother,'split':mother_split[mother],
            'absolute_start':float(start),'absolute_end':float(start+dur),'n_times':len(t),
            'time_min':float(t.min()),'time_max':float(t.max()),'duplicates':0,
            'history_duration':cutoff,'horizon':horizon,'n_history':len(ri),'n_targets':len(qy),
            'missing_fraction':(1-finite.mean(axis=0)).tolist(),
            'target_coverage':len(qy)/(buckets*len(variables))})
    src={str(p):digest(p) for p in source_files}
    protocol={**spec, 'parameters':opts, 'source_hashes':src,'split_mothers':groups}
    version=spec['task_id']+'_'+hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    task=TaskSpec(spec['task_id'],version,Track.NATIVE_IRR,spec['backend'],splits,list(variables),[],
        'first 80% of fixed raw-time window; history includes cutoff',cutoff,horizon,
        QueryPolicy.BUCKETED,(edges[1:]-cutoff).tolist(),metric_version=cfg['evaluation']['metric_version'],f0_model='APN',
        notes=spec['notes'])
    task.validate()
    card=DatasetCard(version,spec['backend'],variables,
        {'unit':spec['time_unit'],'origin':'window start','physical_time_unit':spec['time_unit'],
         'history_duration':cutoff,'availability':'measurement timestamp; daily reports at day end; retrospective data, publication latency unknown',
         'irregularity':spec['irregularity']},provenance=spec['provenance'],unknown=spec.get('unknown',[]))
    bundle=TaskBundle(task,card,EventTable.concat(histories),EventTable.concat(targets),list(variables),anchors,{})
    bundle.data_audit={'status':'prepared','dataset_version':version,'source_files':src,
        'raw_structure':'raw station/subject source -> fixed disjoint windows -> event tables',
        'variable_mapping':variables,'time_semantics':card.time_semantics,'n_entities':len(records),
        'n_mothers':len(mothers),'split_mothers':groups,'split_ids':splits,'records':audit_rows,
        'normalization':'none in loader; Harness fits training-history statistics only',
        'interpolation':'none; absent/NaN observations omitted, never zero-filled as observations',
        'protocol':protocol,'task_spec':asdict(task)}
    return bundle
