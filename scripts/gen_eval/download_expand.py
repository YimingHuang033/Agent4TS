"""Resume downloads; verify pinned bytes before atomic rename. No dependencies."""
import hashlib,json,subprocess,sys
from pathlib import Path
from agent4ts import config
from agent4ts.data.raw_forecast import digest
cfg=config.load_config('gen_eval/main',extra=['gen_eval/expand_v3']);opts=cfg['expansion'];raw=Path(opts['raw_root'])
lock_path=Path(cfg['paths']['resource_lock']);lock=json.loads(lock_path.read_text()).get('expand_1109',{})
records=[]
def fetch(url,path,sha=None,size=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (not sha or digest(path)==sha) and (not size or path.stat().st_size==size):return
    part=path.with_name(path.name+'.download.part')
    subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','-C','-','-o',str(part),url],check=True)
    actual=digest(part)
    if sha and actual!=sha:raise RuntimeError(f'SHA mismatch {path}: {actual}')
    if size and part.stat().st_size!=size:raise RuntimeError(f'Byte count mismatch {path}')
    part.replace(path);records.append({'url':url,'path':str(path),'sha256':actual,'bytes':path.stat().st_size})
for name,item in opts['snapshots'].items():
    path=raw/item['relative_path']
    try:fetch(item['url'],path,item['sha256'],item['size'])
    except subprocess.CalledProcessError:
        fetch(item['url'].replace('https://huggingface.co','https://hf-mirror.com'),path,item['sha256'],item['size'])
ushcn=Path(cfg['paths']['ushcn_root'])
fetch(opts['sources']['ushcn_metadata'],ushcn/'ghcnd-stations.txt',opts['ushcn_metadata_sha256'])
selection=ushcn/'selection.json'
if not selection.exists():
    ids=sorted(l[:11] for l in (ushcn/'ghcnd-stations.txt').read_text().splitlines() if l[76:79]=='HCN')
    selection.write_text(json.dumps({'rule':'20 equally spaced indices in sorted NOAA HCN-flag station IDs; no values inspected','station_ids':[ids[round(i*(len(ids)-1)/19)] for i in range(20)]},indent=2))
for station in json.loads(selection.read_text())['station_ids']:
    path=ushcn/(station+'.dly')
    fetch(opts['sources']['ushcn_station_base']+path.name,path,lock.get('raw_files',{}).get(str(path)))
subset=Path(opts['pamap_subset'])
if not subset.exists() or digest(subset)!=opts['pamap_subset_sha256']:
    archive=Path(opts['pamap_zip'])
    fetch(opts['sources']['pamap2_zip'],archive,opts['pamap_archive_sha256'])
    subprocess.run([sys.executable,'scripts/gen_eval/subset_pamap.py',str(archive),str(subset)],check=True)
    # Zip container timestamp can vary; compare content hashes instead of silently accepting a new version.
    if digest(subset)!=opts['pamap_subset_sha256']:raise RuntimeError('Rebuilt subset differs; verify member hashes and deliberately version config before use')
out=Path(opts['audit_root']);out.mkdir(parents=True,exist_ok=True)
if records:(out/'downloads_latest.json').write_text(json.dumps(records,indent=2))
print('Pinned snapshots and raw forecast resources verified')
