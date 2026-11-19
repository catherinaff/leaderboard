"""Fetch reviewed author sources at pinned versions, preserving SOURCE.json."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.py','.md','.json','.yaml','.yml','.sh','.toml','.cfg','.ini','.ipynb','.txt'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def git(folder,*args):
    return subprocess.check_output(['git','-C',str(folder),*args],stderr=subprocess.STDOUT,text=True).strip()


def fetch(item):
    dest=ROOT/item['destination'];dest.mkdir(parents=True,exist_ok=True)
    meta=item['metadata']
    if meta.get('pinned_url') and meta.get('archive_sha256'):
        request=urllib.request.Request(meta['pinned_url'],headers={'User-Agent':'IoTBench-reproduction'})
        with urllib.request.urlopen(request,timeout=120) as stream:
            raw=stream.read(64*1024*1024+1)
        assert len(raw)<=64*1024*1024
        assert hashlib.sha256(raw).hexdigest()==meta['archive_sha256'],'Source archive checksum mismatch'
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                relative=Path(*Path(member.filename).parts[1:])
                if member.is_dir() or not relative.parts:continue
                assert not relative.is_absolute() and '..' not in relative.parts
                if relative.suffix.lower() not in EXTENSIONS and not relative.name.upper().startswith(('LICENSE','COPYING','NOTICE')):continue
                if member.file_size>8*1024*1024:continue
                output=dest/relative;output.parent.mkdir(parents=True,exist_ok=True)
                body=archive.read(member)
                if output.exists() and output.read_bytes()!=body:
                    raise RuntimeError(f'Refusing to overwrite a modified upstream source: {output}')
                output.write_bytes(body)
    else:
        # Git metadata is retained because the TensorFlow runner verifies
        # git-show HEAD:model.py against its loaded author model file.
        if not (dest/'.git').exists():
            subprocess.run(['git','init',str(dest)],check=True,stdout=subprocess.DEVNULL)
            git(dest,'remote','add','origin',meta['repo'])
            git(dest,'config','core.sparseCheckout','true')
            git(dest,'config','remote.origin.promisor','true')
            git(dest,'config','remote.origin.partialclonefilter','blob:none')
            excludes=['*.pt','*.pth','*.bin','*.safetensors','*.npy','*.npz','*.pkl','*.zip','*.gz','*.pdf','*.h5','*.hdf5','*.png','*.jpg','*.mp4','*.csv','*.mat','*.dat','*.wav','*.flac','*.xlsx','*.xls']
            sparse='/*\n'+''.join('!'+x+'\n' for x in excludes)
            (dest/'.git/info/sparse-checkout').write_text(sparse)
        else:
            assert git(dest,'remote','get-url','origin').rstrip('/').removesuffix('.git')==meta['repo'].rstrip('/').removesuffix('.git')
            if git(dest,'status','--porcelain','--untracked-files=no'):
                raise RuntimeError(f'Upstream checkout has local changes: {dest}')
        git(dest,'fetch','--depth=1','--filter=blob:none','origin',meta['commit'])
        git(dest,'checkout','--detach','FETCH_HEAD')
        assert git(dest,'rev-parse','HEAD').startswith(meta['commit'])
    mapping=meta.get('files_sha256') or meta.get('source_files') or {}
    for name,digest in mapping.items():
        if isinstance(digest,dict):digest=digest['sha256']
        assert sha(dest/name)==digest,('Upstream file mismatch',name)
    (dest/'SOURCE.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('verified',item['name'],meta['commit'])


def link_sources():
    old=ROOT/'experiments/imu_reproduction_20261028/source'
    target=ROOT/'experiments/imu_continuation_20261106/source'
    if not target.exists():
        target.symlink_to(Path('../imu_reproduction_20261028/source'),target_is_directory=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--names',nargs='+');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();items=json.loads((ROOT/'configs/upstream_sources.json').read_text())['sources']
    if a.names:
        missing=set(a.names)-{x['name'] for x in items}
        if missing:p.error('Unknown source names: '+', '.join(sorted(missing)))
        items=[x for x in items if x['name'] in a.names]
    elif not a.all and not a.dry_run:p.error('Choose --all or --names; --dry-run prints the complete plan')
    if a.dry_run:
        print(json.dumps([{'name':x['name'],'destination':x['destination'],'repo':x['metadata']['repo'],'commit':x['metadata']['commit']} for x in items],indent=2));return
    for item in items:fetch(item)
    link_sources()


if __name__=='__main__':main()
