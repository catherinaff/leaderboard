"""Publish this training job's report once per minute; no general shell access."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent


def status(value):
    path=HERE/'publisher.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**value},indent=2)+'\n')
    tmp.replace(path)


def controller_alive():
    pid=json.loads((HERE/'controller.json').read_text())['pid']
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
        state=Path(f'/proc/{pid}/stat').read_text().split()[2]
        return str(HERE/'controller.py').encode() in cmd and state!='Z'
    except FileNotFoundError:return False

if __name__=='__main__':
    lock=(HERE/'publisher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=['ssh','-T','-p','46429','-i',str(HERE/'report_ed25519'),'-o','BatchMode=yes',
         '-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o',
         'UserKnownHostsFile='+str(HERE/'server16_known_hosts'),'root@10.130.151.16']
    down_since=None
    while True:
        try:
            p=subprocess.run(cmd,input=(HERE/'report.json').read_bytes(),capture_output=True,timeout=20)
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),p.returncode,p.stdout.decode().strip(),p.stderr.decode()[-400:],flush=True)
            status({'state':'published' if p.returncode==0 else 'retrying','exit_code':p.returncode,
                    'receipt':p.stdout.decode().strip(),'error':p.stderr.decode()[-400:]})
            if controller_alive():down_since=None
            else:
                down_since=down_since or time.monotonic()
                # A controller replacement must not silently stop publication.
                if time.monotonic()-down_since>=120 and p.returncode==0:
                    status({'state':'finished','receipt':p.stdout.decode().strip()})
                    print('controller finished; final report published',flush=True);break
        except Exception as e:
            print(type(e).__name__,str(e),flush=True)
            status({'state':'retrying','error':f'{type(e).__name__}: {e}'})
        time.sleep(60)
