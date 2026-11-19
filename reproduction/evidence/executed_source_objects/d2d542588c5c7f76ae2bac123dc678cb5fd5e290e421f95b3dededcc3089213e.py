"""Preserve the failed HHAR artifacts and publish only a verified full retry."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'imu_reproduction_20261028'
REL=Path('runs/robusthar/hhar/seed_3407/test_3_val_4')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state,**extra):
    file=HERE/'state/previous_repair.json';tmp=file.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=state,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n');tmp.replace(file)


def main():
    lock=(HERE/'previous_repair.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=HERE/REL/'result.json'
    while not result.exists():
        status('waiting_for_complete_robusthar_retry');time.sleep(30)
    row=json.loads(result.read_text())
    assert row['status']=='complete' and (row['method'],row['dataset'],row['seed'],row['fold'])==('robusthar','hhar',3407,'test_3_val_4')
    assert row['recipe']['sdpa_backend']=='math'
    assert row['dataset_manifest_sha256']==sha(OLD/'data/hhar/manifest.json')
    for name,digest in row['artifacts'].items():assert sha(result.parent/name)==digest
    with np.load(result.parent/'predictions.npz') as pred:
        assert len(pred['y_true'])==row['counts']['test']>0
        assert np.isfinite(pred['logits']).all()
        assert np.array_equal(pred['y_pred'],pred['logits'].argmax(1))
    source=result.parent/'execution_source/train.py'
    assert b'error_if_nonfinite=True' in source.read_bytes()
    out=OLD/REL
    archive=OLD/'archive/robusthar_nonfinite_before_math_retry_20261106'/REL.relative_to('runs')
    if not out.is_symlink():
        assert not (out/'result.json').exists(),'Unexpected completed run at original destination'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            assert not archive.exists();out.rename(archive)
        out.symlink_to(result.parent,target_is_directory=True)
    assert out.resolve()==result.parent.resolve()
    failures=json.loads((OLD/'failures.json').read_text())
    backup=HERE/'archives/failures_before_robust_repair.json'
    if not backup.exists():backup.write_text(json.dumps(failures,indent=2)+'\n')
    keys=[key for key,v in failures.items() if (v['method'],v['dataset'],v['seed'],v['fold'])==('robusthar','hhar',3407,'test_3_val_4')]
    for key in keys:failures.pop(key)
    (OLD/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    code="import controller,json;from pathlib import Path;controller.report({},json.loads(Path('failures.json').read_text()))"
    subprocess.run([sys.executable,'-c',code],cwd=OLD,check=True)
    report=json.loads((OLD/'report.json').read_text())
    cell=next(c for c in report['cells'] if c['method']=='robusthar' and c['dataset']=='hhar')
    assert cell['status']=='complete' and cell['completed_folds']==45
    cmd=['ssh','-T','-p','46429','-i',str(OLD/'report_ed25519'),'-o','IdentitiesOnly=yes',
        '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
        '-o','UserKnownHostsFile='+str(OLD/'server16_known_hosts'),'root@10.130.151.16']
    while True:
        published=subprocess.run(cmd,input=(OLD/'report.json').read_bytes(),capture_output=True,timeout=45)
        if not published.returncode:
            status('complete_and_published',failed_artifacts_archive=str(archive),replacement_result_sha256=sha(result),receipt=published.stdout.decode())
            print('Verified RobustHAR 45/45 HHAR runs; archived failed artifacts; published complete aggregate',flush=True)
            return
        status('retrying_report_publication',error=published.stderr.decode(errors='replace')[-1000:]);time.sleep(60)


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=f'{type(e).__name__}: {e}')
        raise
