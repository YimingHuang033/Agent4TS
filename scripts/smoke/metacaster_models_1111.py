"""CPU adapter check using an existing, independently validated MGAgent artifact.

No new LLM request: this is a model-interface regression, not a formal result.
"""
import os,json,hashlib
from pathlib import Path
import numpy as np
from agent4ts.config import load_config
from agent4ts.data import load_task
from agent4ts.baselines import metacaster
from agent4ts import harness
_original_make=harness.make_model
def one_epoch(mid,settings,resources):
    model=_original_make(mid,settings,resources)
    if hasattr(model,"epochs"):model.epochs=1
    return model
harness.make_model=one_epoch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
cfg=load_config('gen_eval/main',extra=['gen_eval/expand_v3','gen_eval/metacaster_1111'])
cfg['metacaster'].update(support_size=30,n_target=64)
source=Path('results/smoke/metacaster_1111/physiome_dupont/seed0/metacaster/attempt001')
assert json.loads((source/'mgagent/independent_validation.json').read_text())['passed']
expected=json.loads((source/'support.json').read_text())['tensor_sha256']
def replay(samples,cfg,workspace,seed):
    assert hashlib.sha256(samples.tobytes()).hexdigest()==expected
    return np.load(source/'mgagent/dataset.npy',allow_pickle=False),[]
metacaster.generate=replay
bundle=load_task(cfg['tasks']['physiome_dupont'],cfg)
out=Path('results/smoke/metacaster_models_1111');out.mkdir(exist_ok=False)
r=metacaster.run_metacaster(bundle,cfg,out)
r['evidence_kind']='CPU one-epoch model-interface regression; cached real MGAgent output, no API calls'
(out/'report.json').write_text(json.dumps(r,indent=2,default=str))
assert len(r['archive'])==12
assert all(x['status']=='ok' for x in r['archive']),[(x['pipeline']['model_id'],x['error']) for x in r['archive'] if x['status']!='ok']
assert r['status']=='ok'
print('All 12 model adapters + frozen finalizer passed; GPU hours',r['used']['gpu_hours'])
