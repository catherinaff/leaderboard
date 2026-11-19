"""Publish this training job's report once per minute; no general shell access."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    lock=(HERE/'publisher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=['ssh','-T','-p','46429','-i',str(HERE/'report_ed25519'),'-o','BatchMode=yes',
         '-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o',
         'UserKnownHostsFile='+str(HERE/'server16_known_hosts'),'root@10.130.151.16']
    while True:
        try:
            p=subprocess.run(cmd,input=(HERE/'report.json').read_bytes(),capture_output=True,timeout=20)
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),p.returncode,p.stdout.decode().strip(),p.stderr.decode()[-400:],flush=True)
            pid=json.loads((HERE/'controller.json').read_text())['pid']
            os.kill(pid,0)
        except ProcessLookupError:
            print('controller finished; final report published',flush=True);break
        except Exception as e:print(type(e).__name__,str(e),flush=True)
        time.sleep(60)
