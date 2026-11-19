"""Finish the source-input correction audit after all ten formal reruns exist."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SEEDS=[42,123,2021,2024,3407]


def write(path,value):
    temp=path.with_name(path.name+'.input-audit-tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def state(status,**extra):
    value=dict(status=status,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**extra)
    write(HERE/'state/input_fix_verification.json',value)


def main():
    lock=(HERE/'input_fix_verification.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[HERE/'runs'/method/'uci_har'/f'seed_{seed}'/'official/result.json' for method in ['hart','mobilehart'] for seed in SEEDS]
    while not all(file.exists() for file in files):
        state('waiting_for_all_ten_corrected_runs',complete=sum(f.exists() for f in files),required=10)
        time.sleep(30)
    from audit_results_20261111 import verify
    verified=[verify(file) for file in files]
    from controller import summarize,read
    corrections=read(HERE/'state/corrections.json',[])
    for entry in corrections:
        if entry['id']!='uci-hart-total-acc-20261111':continue
        summary={}
        for method in entry['methods']:
            rows=[json.loads(file.read_text()) for file in files if json.loads(file.read_text())['method']==method]
            assert len(rows)==5
            metrics={}
            for metric in ['accuracy','f1_macro','f1_weighted']:
                values=[row['test'][metric] for row in rows];avg=sum(values)/5
                metrics[metric]=avg;metrics[metric+'_std']=(sum((v-avg)**2 for v in values)/5)**.5
            summary[method]=metrics
        entry.update(status='rerun_complete',verification='Ten saved predictions, checkpoints, frozen splits and corrected scaler channel indices independently verified.',corrected_metrics=summary)
    write(HERE/'state/corrections.json',corrections)
    write(HERE/'audits/uci_input_fix_20261111.json',dict(verified_at=datetime.now(timezone.utc).isoformat(),runs=verified,corrections=corrections))
    state('passed',complete=10,required=10,report=str(HERE/'audits/uci_input_fix_20261111.json'))
    print('Ten corrected UCI input reruns independently verified',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('requires_repair',error=f'{type(error).__name__}: {error}')
        raise
