"""CPU adapter integration validation: fit, frozen reload, label independence."""
import json, pickle, copy, os
from pathlib import Path
import numpy as np
from agent4ts import config
from agent4ts.data import load_task
from agent4ts.harness import Harness
from agent4ts.schemas import PipelineSpec,DataView
cfg=config.load_config('smoke/quick');cfg['resources']['upstream_training_epochs']=1
cfg['tasks']['physiome_dupont']['n_entities']=10
bundle=load_task(cfg['tasks']['physiome_dupont'],cfg)
age = os.environ.get('ATS_CHECK_AGE') == '1'
view=DataView(age_feature=age)
root=Path(cfg['paths']['results_root'])/'smoke'/('upstream_age_checks' if age else 'upstream_adapter_checks');rows=[]
from agent4ts.models.registry import runnable_ids,CATALOG
for mid in runnable_ids():
    if age and not CATALOG[mid].get("supports_age_features",False):continue
    h=Harness(bundle,cfg,root/mid);ps=PipelineSpec('adapter',view,mid,{},0)
    fb=h.run_trial(ps);row={'model':mid,'status':fb.status,'error':fb.error}
    if fb.status=='ok':
        payload=h.export(ps,root/mid/'export')
        model=pickle.loads((root/mid/'export'/payload['model_artifact']).read_bytes())
        lw=h.windows('search',view)[0];before=model.predict(lw)
        other=copy.deepcopy(lw);other.y_true[:]=999999;other.y_scored[:]=False
        np.testing.assert_allclose(before,model.predict(other))
        assert np.isfinite(before).all()
        row['frozen_reload_and_label_independence']='passed'
    rows.append(row);print(json.dumps(row),flush=True)
(root/'report.json').write_text(json.dumps(rows,indent=2))
