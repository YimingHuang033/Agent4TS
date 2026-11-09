"""Read-only progress for the versioned experiment matrix."""
from pathlib import Path
from collections import Counter
import json
root=Path(__file__).resolve().parents[2]/'results/gen_eval/v2_ark_20261107'
rows=[]
for path in root.rglob('result.json'):
    try:rows.append(json.loads(path.read_text()))
    except (ValueError,OSError):pass
print(json.dumps({'completed_cells':len(rows),'planned_cells':720,
 'statuses':dict(Counter(r['status'] for r in rows)),
 'done_marker':(root/'DONE').exists(),
 'last_cells':[{k:r.get(k) for k in ['task','seed','method','degradation','status']} for r in rows[-5:]]},indent=2))
