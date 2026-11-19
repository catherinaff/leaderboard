"""Publish progress through a report-only forced SSH command on server 16."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent


def main():
    lock=(HERE/'publisher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    while True:
        try:
            raw=(HERE/'report.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=previous:
                cmd=['ssh','-T','-i',str(HERE/'report_ed25519'),'-o','IdentitiesOnly=yes',
                    '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                    '-o','UserKnownHostsFile='+str(HERE/'server16_known_hosts'),'-p','46429','root@10.130.151.16']
                result=subprocess.run(cmd,input=raw,capture_output=True,timeout=45)
                status=dict(pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),
                    status='published' if result.returncode==0 else 'error',report_sha256=digest,
                    response=result.stdout.decode(errors='replace')[:1000],error=result.stderr.decode(errors='replace')[:2000])
                target=HERE/'state/publisher.json';tmp=target.with_suffix('.tmp')
                tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(target)
                print(status,flush=True)
                if result.returncode==0:previous=digest
        except Exception as e:print(type(e).__name__,str(e),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
