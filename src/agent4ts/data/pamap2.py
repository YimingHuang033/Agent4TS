"""Official UCI PAMAP2 Protocol sessions; raw missingness, no interpolation."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .raw_forecast import build


def load_pamap2(spec,cfg):
    from .raw_forecast import digest
    import json
    opts=cfg['expansion']; archive=Path(opts['pamap_subset'])
    assert digest(archive)==opts['pamap_subset_sha256'], 'Raw subset hash mismatch'
    duration=opts['pamap_window_seconds'];nwin=opts['pamap_windows_per_subject'];records=[]
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['columns_0based']==[0,2,3,4,21,38]
        assert manifest['window_seconds']==duration and manifest['windows_per_subject']==nwin
        assert len(manifest['subjects'])==9
        for mother,info in sorted(manifest['subjects'].items()):
            for i,start in enumerate(info['starts']):
                with z.open(f'{mother}/{i:02}.csv') as stream: frame=pd.read_csv(stream,header=None)
                t=frame[0].to_numpy(); x=frame.iloc[:,1:].to_numpy()
                records.append((mother,float(start),t-start,x,duration))
    variables={
        'heart_rate':{'unit':'bpm','source_column_1based':3},
        'hand_temperature':{'unit':'degC','source_column_1based':4},
        'hand_acc16_x':{'unit':'m/s^2','source_column_1based':5},
        'chest_acc16_x':{'unit':'m/s^2','source_column_1based':22},
        'ankle_acc16_x':{'unit':'m/s^2','source_column_1based':39}}
    for v in variables.values():v.update(kind='instantaneous',aggregation='mean',missing_meaning='raw NaN; no interpolation')
    meta={'task_id':'pamap2_forecast','backend':'pamap2','time_unit':'seconds',
        'notes':'Custom forecasting: all nine Protocol subjects, up to 20 evenly spaced disjoint 10-second windows each; 8-second history and ten 0.2-second target means. Five fixed sensor channels; subject-isolated splits. Not activity classification or BITS.',
        'irregularity':'raw 100 Hz IMU / approx 9 Hz heart-rate observation masks retained; no invented HR observations',
        'provenance':[{'url':opts['sources']['pamap2_metadata']}],
        'unknown':['Protocol sensor time synchronization error is not independently estimated.']}
    bundle=build(records,variables,meta,cfg,[archive])
    bundle.data_audit['raw_subset_manifest']=manifest
    return bundle
