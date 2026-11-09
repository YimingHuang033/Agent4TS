"""NOAA GHCN-Daily HCN-flag station subset, explicitly separate from BITS."""
from pathlib import Path
import datetime as dt, json
import numpy as np
from .raw_forecast import build


def load_ushcn(spec,cfg):
    root=Path(cfg['paths']['ushcn_root']);opts=cfg['expansion']
    selection=json.loads((root/'selection.json').read_text()); files=[root/(s+'.dly') for s in selection['station_ids']]
    names=['SNOW','SNWD','PRCP','TMAX','TMIN']; records=[]
    y0,y1=opts['ushcn_years'];span=opts['ushcn_window_days']
    origin=dt.date(y0,1,1);end=dt.date(y1+1,1,1);ndays=(end-origin).days
    for path in files:
        x=np.full((ndays,5),np.nan)
        for line in path.read_text().splitlines():
            year=int(line[11:15]);month=int(line[15:17]);name=line[17:21]
            if not y0<=year<=y1 or name not in names:continue
            for d in range(31):
                try:date=dt.date(year,month,d+1)
                except ValueError:continue
                field=line[21+8*d:29+8*d];value=int(field[:5])
                if value==-9999 or field[6]!=' ':continue
                idx=(date-origin).days
                if np.isfinite(x[idx,names.index(name)]):raise ValueError('Duplicate daily observation')
                x[idx,names.index(name)]=value/(10 if name in ('PRCP','TMAX','TMIN') else 1)
        for start in range(0,ndays-span+1,span):
            records.append((path.stem,float(origin.toordinal()+start),np.arange(1,span+1,dtype=float),x[start:start+span],span))
    variables={
        'SNOW':{'unit':'mm','kind':'daily_interval_amount','aggregation':'sum','require_complete_daily':True},
        'SNWD':{'unit':'mm','kind':'state','aggregation':'last'},
        'PRCP':{'unit':'mm','kind':'daily_interval_amount','aggregation':'sum','require_complete_daily':True},
        'TMAX':{'unit':'degC','kind':'daily_maximum','aggregation':'max','require_complete_daily':True},
        'TMIN':{'unit':'degC','kind':'daily_minimum','aggregation':'min','require_complete_daily':True}}
    for v in variables.values():v['missing_meaning']='NOAA -9999 or nonblank quality flag; excluded'
    meta={'task_id':'ushcn_forecast','backend':'ushcn','time_unit':'days',
        'notes':'Custom forecasting: 20 prespecified HCN stations, 1990-1999; 100-day disjoint windows, 80-day history, ten 2-day query buckets. Not BITS scores. Mother station splits.',
        'irregularity':'daily calendar with naturally missing or rejected quality-flag measurements; no synthetic subsampling',
        'provenance':[{'url':opts['sources']['ushcn_readme'],'selection':selection}],
        'unknown':['Daily publication delay unknown; retrospective daily availability convention.']}
    return build(records,variables,meta,cfg,files+[root/'selection.json',root/'ghcnd-stations.txt'])
