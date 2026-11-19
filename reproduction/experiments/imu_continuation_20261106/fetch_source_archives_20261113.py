"""Acquire immutable author snapshots through GitHub's archive API."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import io
import json
import re
import time
import zipfile
import requests

HERE=Path(__file__).resolve().parent
REPOS={
 'gpt4ts':'DAMO-DI-ML/NeurIPS2023-One-Fits-All',
 'units':'mims-harvard/UniTS','calf':'Hank0626/CALF','timellm':'KimMeen/Time-LLM',
 'imu2clip':'facebookresearch/imu2clip','onellm':'csuhan/OneLLM',
 'sensorllm':'CRUISEResearchGroup/SensorLLM','imugpt':'ZikangLeng/IMUGPT',
 'bymyeyes':'diamond264/ByMyEyes',
}
EXTENSIONS={'.py','.md','.json','.yaml','.yml','.sh','.toml','.cfg','.ini','.ipynb'}


def main():
 session=requests.Session();manifest={};directory=HERE/'vendor';directory.mkdir(exist_ok=True)
 for name,repo in REPOS.items():
  dest=directory/name;dest.mkdir(exist_ok=True)
  if (dest/'SOURCE.json').exists():
   info=json.loads((dest/'SOURCE.json').read_text())
   if info.get('provider')=='GitHub immutable source archive':
    manifest[name]=info;print(name,'already acquired',flush=True);continue
  try:
   # Codeload avoids the shared unauthenticated API rate limit. Legacy
   # archive roots identify the commit, and SHA256 pins the exact bytes.
   response=session.get(f'https://codeload.github.com/{repo}/legacy.zip/refs/heads/main',stream=True,timeout=25)
   if response.status_code==404:
    response.close();response=session.get(f'https://codeload.github.com/{repo}/legacy.zip/refs/heads/master',stream=True,timeout=25)
   response.raise_for_status()
   chunks=[];size=0
   for chunk in response.iter_content(1024*1024):
    size+=len(chunk)
    if size>64*1024*1024:raise ValueError('Repository archive exceeds 64 MiB source-download cap; needs a selective fetch')
    chunks.append(chunk)
   raw=b''.join(chunks);files={};notebooks=[]
   with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    root=archive.namelist()[0].split('/')[0];match=re.search(r'-([0-9a-f]{7,40})$',root)
    if not match:raise ValueError('Source archive does not identify its commit')
    commit=match[1]
    for member in archive.infolist():
     parts=Path(member.filename).parts[1:]
     if member.is_dir() or not parts or '..' in parts:continue
     rel=Path(*parts)
     keep=rel.suffix.lower() in EXTENSIONS or rel.name.upper().startswith('LICENSE') or (rel.suffix.lower()=='.txt' and member.file_size<128*1024)
     if not keep or member.file_size>8*1024*1024:continue
     content=archive.read(member);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
     files[str(rel)]=hashlib.sha256(content).hexdigest()
     if rel.suffix.lower()=='.ipynb':notebooks.append(str(rel))
   info=dict(provider='GitHub immutable source archive',repo='https://github.com/'+repo,commit=commit,commit_is_abbreviated=len(commit)<40,pinned_url=f'https://codeload.github.com/{repo}/legacy.zip/{commit}',archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=size,
     files_sha256=files,notebooks=notebooks,fetched_at=datetime.now(timezone.utc).isoformat(),status='source_ready')
   (dest/'SOURCE.json').write_text(json.dumps(info,indent=2)+'\n')
   print(name,commit,'ready',len(files),'source files',round(size/2**20,2),'MiB archive',flush=True)
  except Exception as error:
   info=dict(repo='https://github.com/'+repo,status='fetch_failed',error=f'{type(error).__name__}: {str(error)[:300]}')
   print(name,info['error'],flush=True)
  manifest[name]=info
  temp=HERE/'source_acquisition_20261113.json.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(HERE/'source_acquisition_20261113.json')


if __name__=='__main__':main()
