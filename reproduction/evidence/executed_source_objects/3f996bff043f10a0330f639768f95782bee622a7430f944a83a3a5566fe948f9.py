"""Resumable large public downloads with fresh URLs and original SHA checks."""
from pathlib import Path
import hashlib
import json
import shutil
import time

import requests
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
API=HfApi(endpoint='https://hf-mirror.com',token=False)


def sha(file):
 h=hashlib.sha256()
 with file.open('rb') as source:
  for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()


def download(repo,revision,name,directory,expected_sha=None,size=None):
 target=directory/name;part=target.with_suffix(target.suffix+'.part')
 if target.exists():
  if (not size or target.stat().st_size==size) and (not expected_sha or sha(target)==expected_sha):return target
  raise ValueError(f'Existing complete asset differs: {name}')
 if not part.exists() and expected_sha:
  candidates=list((directory/'.cache/huggingface/download').glob(f'*.{expected_sha}.incomplete'))
  if len(candidates)==1:candidates[0].replace(part)
 for attempt in range(1,7):
  if shutil.disk_usage(directory).free<36*2**30:raise RuntimeError('Maintain disk reserve for outputs')
  offset=part.stat().st_size if part.exists() else 0
  if size and offset==size:break
  url=f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}'
  try:
   response=requests.get(url,params={'download':'true','refresh':str(time.time_ns())},
       headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(25,120))
   response.raise_for_status()
   if offset:
    if response.status_code!=206:raise RuntimeError('Server must honor Range to preserve an existing partial asset')
    if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Unexpected byte-range offset')
   received=offset;last_log=offset
   with part.open('ab' if offset else 'wb') as output:
    for block in response.iter_content(1024*1024):
     output.write(block);received+=len(block)
     if size and received>size:raise ValueError('Asset exceeds expected original size')
     if received-last_log>=1024*1024*1024:
      print(repo,name,'bytes',received,'of',size,flush=True);last_log=received
   if size and part.stat().st_size!=size:raise ValueError('Truncated asset')
   break
  except Exception as error:
   # Avoid writing temporary CDN authorization URLs into logs.
   print(repo,name,'attempt',attempt,type(error).__name__,'resume_bytes',part.stat().st_size if part.exists() else 0,flush=True)
   if attempt==6:raise RuntimeError(f'Public download remains unavailable: {repo}/{name}') from None
   time.sleep(5)
 if size:assert part.stat().st_size==size
 if expected_sha:assert sha(part)==expected_sha,f'Original asset SHA mismatch: {name}'
 part.replace(target);return target


def model(repo,name,files,official=None):
 info=API.model_info(repo,files_metadata=True,timeout=30)
 original=API.model_info(official,files_metadata=True,timeout=30) if official else info
 values={f.rfilename:f for f in info.siblings};originals={f.rfilename:f for f in original.siblings}
 directory=HERE/'assets_20261113'/name;directory.mkdir(exist_ok=True)
 manifest=dict(repo=repo,revision=info.sha,official_repo=original.id,official_revision=original.sha,files={})
 for file in files:
  value=values[file];expected=value.lfs.sha256 if value.lfs else None
  if official and value.lfs:
   assert originals[file].lfs.sha256==expected and originals[file].size==value.size
  target=download(repo,info.sha,file,directory,expected,value.size)
  manifest['files'][file]=dict(sha256=sha(target),size=target.stat().st_size)
  print('VERIFIED',name,file,flush=True)
 (directory/'ASSETS.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
 for name,repo,files,official in [
  ('onellm','csuhan/OneLLM-7B',['params.json','tokenizer.model','consolidated.00-of-01.pth'],None),
  ('llama3','NousResearch/Meta-Llama-3-8B',['config.json','generation_config.json','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json',*[f'model-0000{i}-of-00004.safetensors' for i in range(1,5)]],'meta-llama/Meta-Llama-3-8B')]:
  try:model(repo,name,files,official)
  except Exception as error:print('FAILED',name,type(error).__name__,str(error)[:180],flush=True)
