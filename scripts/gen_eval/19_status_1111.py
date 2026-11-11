"""Read-only overview. Phase memberships overlap; unique cells counted once."""
import json
from collections import Counter
from pathlib import Path
root=Path('results/gen_eval/v3_expand_20261109')
manifest=json.loads((root/'manifest.json').read_text())
unique=Counter();phases={};unfinished=[]
for cell in manifest['cells']:
    attempts=sorted((root/'cells'/cell['id']).glob('attempt*/result.json'))
    status=json.loads(attempts[-1].read_text())['status'] if attempts else 'pending'
    unique[status]+=1
    for phase in cell['phases']:phases.setdefault(phase,Counter())[status]+=1
    if status!='ok':unfinished.append({'id':cell['id'],'status':status,'phases':cell['phases'],'identity':cell['identity']})
print(json.dumps({'unique':unique,'phases':phases,'unfinished_count':len(unfinished)},indent=2))
for base in sorted(Path('results/gen_eval').glob('metacaster_20261111*')):
    counts=Counter()
    for cell in base.glob('*/seed*/*'):
        attempts=sorted(cell.glob('attempt*/result.json'))
        if attempts:counts[json.loads(attempts[-1].read_text())['status']]+=1
    print(base,json.dumps(counts))
