from pathlib import Path
import json
from agent4ts import config
from agent4ts.data import load_task
from agent4ts.baselines.tsci_adapted import run_tsci_adapted
cfg=config.load_config('smoke/ark');cfg['resources']['upstream_training_epochs']=1
cfg['tasks']['physiome_dupont']['n_entities']=10
bundle=load_task(cfg['tasks']['physiome_dupont'],cfg)
out=Path(cfg['paths']['results_root'])/'smoke'/'tsci_adapted'
result=run_tsci_adapted(bundle,cfg,out)
(out/'result.json').write_text(json.dumps(result,indent=2,default=str))
print(json.dumps({'status':'ok','used':result['used'],'incumbent':result['incumbent']['model_id']}))
