"""Lossless raw-column/window extraction; no scaling, fill or interpolation."""
import zipfile,json,hashlib,math,sys
from pathlib import Path
src=Path(sys.argv[1]);out=Path(sys.argv[2]);cols=[0,2,3,4,21,38];duration=10.;nwin=20
manifest={'source_url':'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'columns_0based':cols,'window_seconds':duration,'windows_per_subject':nwin,'subjects':{}}
def write_member(dest,name,content):
 info=zipfile.ZipInfo(name,date_time=(2026,11,9,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
 dest.writestr(info,content)
with zipfile.ZipFile(src) as z, zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as dest:
 for name in sorted(n for n in z.namelist() if '/Protocol/subject' in n and n.endswith('.dat')):
  # All raw bytes hashed; only metadata (times) used to select windows.
  sha=hashlib.sha256();first=last=None;count=0
  with z.open(name) as f:
   for line in f:
    sha.update(line); t=float(line.split(None,1)[0]);first=t if first is None else first;last=t;count+=1
  total=int((last-first)/duration);n=min(total,nwin)
  starts=[int(i*(total-1)/(n-1))*duration+first for i in range(n)]
  mother=Path(name).stem;windows=[[] for _ in starts]
  with z.open(name) as f:
   wi=0
   for line in f:
    fields=line.decode().split();t=float(fields[0])
    while wi<len(starts) and t>=starts[wi]+duration:wi+=1
    if wi==len(starts):break
    if t>=starts[wi]:windows[wi].append(','.join(fields[i] for i in cols))
  for i,rows in enumerate(windows):write_member(dest,f'{mother}/{i:02}.csv','\n'.join(rows)+'\n')
  manifest['subjects'][mother]={'member':name,'source_sha256':sha.hexdigest(),'source_rows':count,'source_first':first,'source_last':last,'starts':starts}
  print(mother,count,flush=True)
 write_member(dest,'manifest.json',json.dumps(manifest,indent=2))
print('subset_sha256',hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)
