"""Verified public SensorLLM assets and noninteractive IMUGPT asset retry."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import HfApi,hf_hub_download
import requests

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets_20261113'


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()


def chronos():
 p=ASSETS/'chronos';p.mkdir(exist_ok=True)
 if (p/'ASSETS.json').exists():return
 info=HfApi(endpoint='https://hf-mirror.com').model_info('amazon/chronos-t5-large',files_metadata=True)
 names={f.rfilename:f for f in info.siblings}
 weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
 manifest=dict(repo=info.id,revision=info.sha,files={})
 for name in ['config.json',weight]:
  if shutil.disk_usage(p).free<40*2**30:raise RuntimeError('Keep 40 GiB free while fetching assets')
  path=Path(hf_hub_download(info.id,name,revision=info.sha,local_dir=p,endpoint='https://hf-mirror.com'))
  digest=sha(path)
  if names[name].lfs:assert digest==names[name].lfs.sha256
  manifest['files'][name]=dict(size=path.stat().st_size,sha256=digest)
  print('Chronos verified',name,flush=True)
 (p/'ASSETS.json').write_text(json.dumps(manifest,indent=2)+'\n')


def mhealth():
 p=ASSETS/'mhealth';p.mkdir(exist_ok=True)
 if (p/'ASSETS.json').exists():return
 url='https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip'
 archive=p/'mhealth.zip'
 if not archive.exists():
  response=requests.get(url,stream=True,timeout=(20,120));response.raise_for_status()
  with archive.with_suffix('.part').open('wb') as target:
   n=0
   for block in response.iter_content(1024*1024):
    n+=len(block)
    if n>100*1024*1024:raise ValueError('Unexpected MHealth archive size')
    target.write(block)
  archive.with_suffix('.part').replace(archive)
 files={}
 with zipfile.ZipFile(archive) as source:
  for member in source.infolist():
   if not member.filename.endswith('.log'):continue
   assert '..' not in Path(member.filename).parts
   file=p/Path(member.filename).name;file.write_bytes(source.read(member))
   files[file.name]=dict(size=file.stat().st_size,sha256=sha(file))
 assert len(files)==10
 (p/'ASSETS.json').write_text(json.dumps(dict(source=url,license='CC BY 4.0',citation='Banos, Garcia & Saez, UCI MHEALTH, doi:10.24432/C5TW22',archive_sha256=sha(archive),files=files),indent=2)+'\n')
 print('MHealth 10 subject logs verified',flush=True)


def imugpt():
 p=ASSETS/'imugpt';p.mkdir(exist_ok=True)
 # Preserve original author normalization arrays and inspect available binary
 # assets from the exact source archive, not from an unverified third party.
 info=json.loads((HERE/'vendor/imugpt/SOURCE.json').read_text())
 response=requests.get(info['pinned_url'],timeout=(20,120));response.raise_for_status()
 assert hashlib.sha256(response.content).hexdigest()==info['archive_sha256']
 manifest=dict(source=info['pinned_url'],archive_sha256=info['archive_sha256'],files={},not_downloaded=['SMPL body-model files: not required by the author inverse-kinematics route.'])
 with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
  for member in archive.infolist():
   relative=Path(*Path(member.filename).parts[1:])
   if member.is_dir() or '..' in relative.parts:continue
   if relative.suffix.lower() not in ['.npy','.npz','.bvh','.pyx','.pxd','.h','.c','.cpp']:continue
   if member.file_size>16*1024*1024:continue
   file=p/'source_data'/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(archive.read(member))
   manifest['files'][str(relative)]=dict(size=file.stat().st_size,sha256=sha(file))
 (p/'SOURCE_ASSETS.json').write_text(json.dumps(manifest,indent=2)+'\n')
 isolated=HERE/'runtime_onellm_20261113/bin/python'
 subprocess.run([str(isolated),'-m','pip','install','gdown==5.2.0'],check=True)
 file=p/'VQTrans_pretrained.zip'
 if not file.exists():
  subprocess.run([str(isolated),'-c',"import gdown,sys;result=gdown.download(id='1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE',output=sys.argv[1],quiet=False,use_cookies=False,resume=True);sys.exit(0 if result else 1)",str(file)],check=True,timeout=1800)
 files={}
 with zipfile.ZipFile(file) as archive:
  for member in archive.infolist():
   rel=Path(member.filename)
   if member.is_dir():continue
   assert not rel.is_absolute() and '..' not in rel.parts
   target=p/'pretrained'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(member))
   files[str(rel)]=dict(size=target.stat().st_size,sha256=sha(target))
 (p/'MOTION_ASSETS.json').write_text(json.dumps(dict(source='Author download_model.sh Google Drive file 1LaOvwypF-jM2Axnq5dc-Iuvv3w_G-WDE',archive_sha256=sha(file),files=files),indent=2)+'\n')
 print('IMUGPT official motion models verified',flush=True)


if __name__=='__main__':
 results={}
 for name,task in [('mhealth',mhealth),('imugpt',imugpt),('chronos',chronos)]:
  try:task();results[name]=dict(status='ready')
  except Exception as error:results[name]=dict(status='unavailable',error=f'{type(error).__name__}: {error}');print(name,results[name]['error'],flush=True)
  (HERE/'remaining_asset_status_20261113.json').write_text(json.dumps(results,indent=2)+'\n')
