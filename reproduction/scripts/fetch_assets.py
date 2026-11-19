"""Fetch explicitly selected, checksum-pinned assets; never call inference APIs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def checked_url(item):
    path=ROOT/item['destination'];path.parent.mkdir(parents=True,exist_ok=True)
    if not (path.exists() and sha(path)==item['sha256']):
        temp=path.with_suffix(path.suffix+'.download')
        request=urllib.request.Request(item['url'],headers={'User-Agent':'IoTBench-reproduction'})
        with urllib.request.urlopen(request,timeout=120) as stream,temp.open('wb') as target:
            shutil.copyfileobj(stream,target,8*1024*1024)
        if sha(temp)!=item['sha256']:
            raise ValueError(f'Asset checksum mismatch; existing target was not replaced: {path}')
        temp.replace(path)
    for alias in item.get('aliases',[]):
        link=ROOT/alias;link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(os.path.relpath(path,link.parent))
    return path


def fetch(name,group):
    manifest=group['manifest'];dest=ROOT/group['destination'];dest.mkdir(parents=True,exist_ok=True)
    if manifest.get('repo') and manifest.get('revision'):
        from huggingface_hub import hf_hub_download
        for filename,info in manifest['files'].items():
            file=Path(hf_hub_download(manifest['repo'],filename,revision=manifest['revision'],local_dir=dest))
            assert sha(file)==info['sha256'],('Asset checksum mismatch',name,filename)
    elif manifest.get('url') and manifest.get('sha256'):
        filename=manifest['url'].rsplit('/',1)[-1]
        checked_url({'destination':str((dest/filename).relative_to(ROOT)),'url':manifest['url'],'sha256':manifest['sha256']})
    else:
        raise ValueError(f'{name} uses the separate original motion/data-generation setup; see method README')
    (dest/'ASSETS.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();groups=json.loads((ROOT/'configs/model_assets.json').read_text());extra=json.loads((ROOT/'configs/extra_assets.json').read_text())
    known=set(groups)|set(extra)
    if set(a.groups)-known:p.error('Unknown asset groups: '+', '.join(sorted(set(a.groups)-known)))
    if a.dry_run:
        print(json.dumps({name:groups[name] if name in groups else extra[name] for name in a.groups},indent=2));return
    for name in a.groups:
        if name in extra:
            for item in extra[name]['files']:print('verified',checked_url(item))
        else:fetch(name,groups[name]);print('verified',name)


if __name__=='__main__':main()
