"""Resume official downloads and prepare RealWorld independently of the SSH client."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=ROOT/'experiments/datasets'
ACTIVITIES=['climbingdown','climbingup','jumping','lying','running','sitting','standing','walking']


def state(status,**values):
    p=HERE/'state/data_worker.json'
    row=dict(status=status,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**values)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(p)


def downloads_running():
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if b'download_realworld_har.py' in proc.read_bytes():return True
        except (FileNotFoundError,PermissionError):pass
    return False


def check():
    missing=[];bad=[]
    for sid in range(1,16):
        for act in ACTIVITIES:
            for sensor in ['acc','gyr']:
                p=RAW/'realworld_har_csv_zips'/f's{sid:02d}'/f'{sensor}_{act}_csv.zip'
                if not p.exists():missing.append(str(p));continue
                try:
                    with zipfile.ZipFile(p) as z:
                        if z.testzip() is not None:bad.append(p)
                except zipfile.BadZipFile:bad.append(p)
    return missing,bad


def main():
    lock=(HERE/'data_worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for attempt in range(7):
        while downloads_running():
            state('waiting_for_existing_download',attempt=attempt)
            time.sleep(20)
        missing,bad=check()
        state('checking_official_archives',missing=len(missing),bad=len(bad),attempt=attempt)
        if not missing and not bad:break
        if attempt==6:raise RuntimeError(f'{len(missing)} missing and {len(bad)} invalid archives after retries')
        for path in bad:
            target=HERE/'archives'/f'bad_{attempt}_{path.parent.name}_{path.name}'
            path.replace(target)
        state('downloading_missing_archives',missing=len(missing),bad=len(bad),attempt=attempt)
        with (HERE/'logs/realworld_download_retry.log').open('ab') as log:
            code=subprocess.call([sys.executable,'-u',str(ROOT/'experiments/download_realworld_har.py'),'--confirm-large','--workers','2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if code:time.sleep(30)
    state('preparing_raw_recording_preserving_windows')
    with (HERE/'logs/realworld_prepare.log').open('ab') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'prepare_extra.py'),'--dataset','realworld_har'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
    state('ready',dataset='realworld_har')


if __name__=='__main__':
    try:main()
    except Exception as e:
        state('failed',error=f'{type(e).__name__}: {e}')
        raise
