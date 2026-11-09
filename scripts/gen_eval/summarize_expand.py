"""Summarize immutable v3 attempts; missing/failed cells never enter means."""
import json,csv
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from run_expand import settings
cfg=settings();root=Path(cfg['expansion']['root']);manifest=json.loads((root/'manifest.json').read_text())
groups=defaultdict(list);failures=[];curves=[];distribution=Counter();costs=[];group_rows=[];behavior=[];finished=True
for cell in manifest['cells']:
    ident=cell['identity'];attempts=sorted((root/'cells'/cell['id']).glob('attempt*/result.json'))
    first=json.loads(attempts[0].read_text()) if attempts else {};last=json.loads(attempts[-1].read_text()) if attempts else {'status':'pending'}
    if last['status'] in ('pending','running'):finished=False
    key=(ident['task'],ident['method'],ident['budget'],ident['f0'],json.dumps(ident['degradation'],sort_keys=True))
    metric=(last.get('test') or {}).get('metrics',{}).get('normalized_mae') if last['status']=='ok' else None
    groups[key].append({'metric':metric,'first_ok':first.get('status')=='ok','status':last['status'],'n_attempts':len(attempts)})
    if metric is None:failures.append({'id':cell['id'],**ident,'status':last['status'],'error':last.get('error')})
    if not attempts:continue
    directory=attempts[-1].parent;trace=directory/'trace.jsonl';best=None;trial=0
    for group,score in (last.get('test') or {}).get('groups',{}).items():
        group_rows.append({'id':cell['id'],'task':ident['task'],'method':ident['method'],'seed':ident['seed'],'budget':ident['budget'],'f0':ident['f0'],'degradation':ident['degradation'],'group':group,**score})
    final_path=directory/'finalize.json'
    if final_path.exists():
        winner=json.loads(final_path.read_text()).get('selected',{})
        distribution[('selected',ident['task'],ident['method'],winner.get('model_id','unknown'))]+=1
    if trace.exists():
        events=[json.loads(line) for line in trace.read_text().splitlines()]
        event_counts=Counter(e.get('event') for e in events)
        proposals=event_counts['propose']
        behavior.append({'id':cell['id'],'proposals':proposals,'duplicates':event_counts['duplicate'],'rejected':event_counts['rejected'],'duplicate_fraction':event_counts['duplicate']/proposals if proposals else None,'stop_reason':next((e.get('reason') for e in reversed(events) if e.get('event')=='stop'),None)})
        for line in trace.read_text().splitlines():
            event=json.loads(line)
            if event.get('event')!='trial' or event.get('split')!='search':continue
            distribution[('visited',ident['task'],ident['method'],event['pipeline']['model_id'])]+=1
            trial+=1;value=event.get('metrics',{}).get('normalized_mae')
            if value is not None:best=value if best is None else min(best,value)
            curves.append({'id':cell['id'],'task':ident['task'],'method':ident['method'],'seed':ident['seed'],'budget':ident['budget'],'f0':ident['f0'],'degradation':ident['degradation'],'trial':trial,'best_search_mae':best})
        for remaining in range(trial+1,ident['budget']+1):curves.append({'id':cell['id'],'task':ident['task'],'method':ident['method'],'seed':ident['seed'],'budget':ident['budget'],'f0':ident['f0'],'degradation':ident['degradation'],'trial':remaining,'best_search_mae':best,'held_after_stop':True})
    costs.append({'id':cell['id'],'used':last.get('used'),'wall_s':last.get('wall_s'),'search_wall_s':last.get('search_wall_s'),'llm_wall_s':last.get('llm_wall_s'),'finalize_wall_s':last.get('finalize_wall_s')})
summary=[]
for key,values in groups.items():
    scores=[v['metric'] for v in values if v['metric'] is not None]
    summary.append(dict(zip(('task','method','budget','f0','degradation'),key),mean=float(np.mean(scores)) if scores else None,std=float(np.std(scores,ddof=1)) if len(scores)>1 else None,n_success=len(scores),n_planned=len(values),n_first_success=sum(v['first_ok'] for v in values),n_attempts=sum(v['n_attempts'] for v in values)))
out=root/'summary';out.mkdir(exist_ok=True)
for name,data in [('summary',summary),('failures',failures),('curves',curves),('costs',costs),('groups',group_rows),('behavior',behavior),('model_distribution',[{'kind':key[0],'task':key[1],'method':key[2],'model':key[3],'count':value} for key,value in distribution.items()])]:
    (out/(name+'.json')).write_text(json.dumps(data,indent=2))
with (out/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
(out/'status.json').write_text(json.dumps({'execution_finished':finished,'accepted':False,'note':'Execution completion does not imply protocol acceptance.'},indent=2))
if curves:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for task in sorted({c['task'] for c in curves}):
        fig,ax=plt.subplots()
        for method in ('agent_full','random_full','greedy_full'):
            rows=[c for c in curves if c['task']==task and c['method']==method and c['budget']==12 and c['f0']=='APN' and c['degradation'] is None]
            xx=sorted({r['trial'] for r in rows});yy=[np.mean([r['best_search_mae'] for r in rows if r['trial']==x and r['best_search_mae'] is not None]) for x in xx]
            if xx:ax.plot(xx,yy,label=method)
        ax.set(xlabel='Candidate training attempt (includes f0)',ylabel='Best search normalized MAE',title=task+' (B=12 descriptive prefix)');ax.legend();fig.savefig(out/(task+'_search_curve.png'));plt.close(fig)
print('Summary:',out,'execution_finished=',finished)
