"""Read-only status; planned, success, failures, interrupted and pending."""
import json
from pathlib import Path
from collections import Counter
from run_expand import settings
cfg=settings();root=Path(cfg['expansion']['root']);path=root/'manifest.json'
if not path.exists():raise SystemExit('No frozen manifest yet; preparation/debug incomplete')
manifest=json.loads(path.read_text());summary={}
for phase in cfg['expansion']['phases']:
    counts=Counter()
    for cell in manifest['cells']:
        if phase not in cell['phases']:continue
        counts['planned']+=1
        attempts=sorted((root/'cells'/cell['id']).glob('attempt*/result.json'))
        status=json.loads(attempts[-1].read_text())['status'] if attempts else 'pending'
        counts[status]+=1
    summary[phase]=dict(counts)
print(json.dumps(summary,indent=2))
