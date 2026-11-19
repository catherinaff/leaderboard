"""Persistent GPU1 queue for large native adapters, entirely local inference.

Never submits HARGPT/ByMyEyes. Failed native validations remain visible, and
are retried only when their adapter code changes. Completed zero-shot and large
candidate results remain separate from the five-seed supervised ranking.
"""
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
STATE=HERE/'state/large_methods_20261113.json'
FOLDS=[('uci_har','official'),('usc_had','test_1_val_2'),('pamap2','test_101_val_102'),('realworld_har','test_1_val_2')]
SEEDS=[42,123,2021,2024,3407]


def now():return datetime.now(timezone.utc).isoformat()


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def write(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(method,file):
    paths=[HERE/file,HERE/'api_adapters.py']
    if method=='onellm':paths+=[HERE/'assets_20261113/onellm/ASSETS.json']
    elif method=='sensorllm':paths+=[HERE/'assets_20261113'/name/'ASSETS.json' for name in ['llama3','chronos','mhealth']]+[HERE/'sensorllm_alignment_20261113/result.json',HERE/'sensorllm_alignment_smoke_20261113/result.json',HERE/'assets_20261113/mhealth_partial_verified/PARTIAL_ASSETS.json']
    elif method=='imugpt':paths+=[HERE/'assets_20261113/imugpt'/name for name in ['MOTION_ASSETS.json','NORMALIZATION_ASSETS.json']]+[HERE/'audits/imugpt_simulator_runtime_20261113.json']
    return hashlib.sha256(json.dumps([(str(p.relative_to(HERE)),sha(p) if p.exists() else None) for p in paths]).encode()).hexdigest()


def alive(info):
    try:
        process=Path('/proc')/str(info['pid'])
        command=(process/'cmdline').read_bytes()
        return (process/'stat').read_text().split()[2]!='Z' and str(HERE).encode() in command and b'_adapter.py' in command
    except (FileNotFoundError,KeyError):return False


def saved_task_result(key,info):
    if key=='onellm_native_smoke':return evidence('onellm')
    if key=='sensorllm_alignment_smoke':return (HERE/'sensorllm_alignment_smoke_20261113/result.json').exists()
    if key=='sensorllm_full_alignment':return (HERE/'sensorllm_alignment_20261113/result.json').exists()
    args=info.get('args',[])
    def arg(name):return args[args.index(name)+1] if name in args else None
    ds=arg('--dataset');fold=arg('--fold');seed=arg('--seed') or '42'
    if key.startswith('imugpt_generate_'):return (HERE/'assets_20261113/imugpt/synthetic'/f'{ds}.json').exists()
    if info['method']=='onellm':return (HERE/'zero_shot_runs/onellm'/ds/f'seed_{seed}'/fold/'result.json').exists()
    if info['method']=='sensorllm':return (HERE/('smoke_large' if '--smoke' in args else 'runs')/'sensorllm'/ds/f'seed_{seed}'/fold/'result.json').exists()
    if info['method']=='imugpt':return (HERE/'smoke/imugpt'/ds/f'seed_{seed}'/fold/'result.json').exists()
    return False


def assets(*names):return all((HERE/'assets_20261113'/name/'ASSETS.json').exists() for name in names)


def capacity():
    if shutil.disk_usage(HERE).free<36*2**30:return False,'waiting for 36 GiB free disk'
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    if available<50*2**20:return False,'waiting for 50 GiB available RAM'
    q=subprocess.run(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return (True,'ready') if q.returncode==0 and int(q.stdout.strip())>43000 else (False,'waiting for exclusive GPU1 capacity')


def update_method(method,**values):
    lock=(HERE/'state/readiness_update.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    value=read(HERE/'readiness.json');value[method].update(values)
    write(HERE/'readiness.json',value)


def evidence(method):
    if method=='onellm':
        return all(any(read(f).get('status')=='smoke_complete' for f in (HERE/'smoke_zero_shot/onellm'/ds/'seed_42').glob('*/result.json')) for ds,_ in FOLDS)
    if method=='sensorllm':return all(list((HERE/'smoke_large/sensorllm'/ds/'seed_42'/fold).glob('result.json')) for ds,fold in FOLDS)
    if method=='imugpt':return all((HERE/'smoke/imugpt'/ds/'seed_42'/fold/'result.json').exists() for ds,fold in FOLDS)


def tasks():
    result=[]
    one=HERE/'runtime_onellm_20261113/bin/python';sensor=HERE/'runtime_sensorllm_20261113/bin/python'
    if assets('onellm') and not evidence('onellm'):
        result.append(('onellm_native_smoke','onellm','onellm_adapter.py',one,['--smoke','--gpu','1']))
    ready_sensor=assets('llama3','chronos') and (assets('mhealth') or (HERE/'assets_20261113/mhealth_partial_verified/PARTIAL_ASSETS.json').exists())
    alignment=HERE/'sensorllm_alignment_20261113'
    alignment_smoke=HERE/'sensorllm_alignment_smoke_20261113'
    if ready_sensor:
        if not (alignment_smoke/'result.json').exists():
            result.append(('sensorllm_alignment_smoke','sensorllm','sensorllm_adapter.py',sensor,['--align','--max-steps','2','--gpu','1']))
        elif not evidence('sensorllm'):
            for ds,fold in FOLDS:
                if not (HERE/'smoke_large/sensorllm'/ds/'seed_42'/fold/'result.json').exists():
                    result.append((f'sensorllm_smoke_{ds}','sensorllm','sensorllm_adapter.py',sensor,['--dataset',ds,'--fold',fold,'--smoke','--gpu','1']))
        elif assets('mhealth') and not (alignment/'result.json').exists():
            result.append(('sensorllm_full_alignment','sensorllm','sensorllm_adapter.py',sensor,['--align','--gpu','1']))
    imu=HERE/'assets_20261113/imugpt'
    if (imu/'MOTION_ASSETS.json').exists() and (imu/'NORMALIZATION_ASSETS.json').exists() and (HERE/'audits/imugpt_simulator_runtime_20261113.json').exists():
        for ds,fold in FOLDS:
            if not (imu/'synthetic'/f'{ds}.npz').exists():
                result.append((f'imugpt_generate_{ds}','imugpt','imugpt_adapter.py',one,['--generate','--dataset',ds,'--gpu','1','--max-prompts','10']))
    if all((imu/'synthetic'/f'{ds}.npz').exists() for ds,_ in FOLDS):
        for ds,fold in FOLDS:
            if not (HERE/'smoke/imugpt'/ds/'seed_42'/fold/'result.json').exists():
                result.append((f'imugpt_smoke_{ds}','imugpt','imugpt_adapter.py',one,['--train','--dataset',ds,'--fold',fold,'--smoke','--epochs','1','--batch-size','4','--gpu','1']))
    # Run frozen OneLLM once per actual held-out subject; do not create five
    # artificial independent fits or copy zero-shot runs into supervised cells.
    if evidence('onellm'):
        for ds,_ in FOLDS:
            for split in sorted((HERE/'manifests/splits'/ds/'seed_42').glob('*.npz')):
                if not (HERE/'zero_shot_runs/onellm'/ds/'seed_42'/split.stem/'result.json').exists():
                    result.append((f'onellm_{ds}_{split.stem}','onellm','onellm_adapter.py',one,['--dataset',ds,'--fold',split.stem,'--gpu','1']))
    if evidence('sensorllm') and (alignment/'result.json').exists():
        for ds,_ in FOLDS:
            for seed in SEEDS:
                for split in sorted((HERE/'manifests/splits'/ds/f'seed_{seed}').glob('*.npz')):
                    if not (HERE/'runs/sensorllm'/ds/f'seed_{seed}'/split.stem/'result.json').exists():
                        result.append((f'sensorllm_{ds}_{seed}_{split.stem}','sensorllm','sensorllm_adapter.py',sensor,['--dataset',ds,'--seed',str(seed),'--fold',split.stem,'--gpu','1']))
    return result


def main():
    lock=(HERE/'large_method_worker_20261113.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    journal=read(STATE,dict(jobs={},started_at=now()));journal.update(pid=os.getpid(),gpu=1,paid_api_calls=0)
    while not (HERE/'STOP_LARGE_20261113').exists():
        adopted=[]
        for key,info in journal['jobs'].items():
            if info.get('status')!='running':continue
            if alive(info):adopted.append(key)
            else:info.update(status='complete' if saved_task_result(key,info) else 'failed',completed_at=now(),adopted_process=True)
        if adopted:
            journal.update(active_job=adopted[0],updated_at=now(),waiting_reason='Adopted existing native process; no duplicate GPU worker')
            write(STATE,journal);time.sleep(30);continue
        journal.pop('active_job',None)
        for method in ['onellm','sensorllm','imugpt']:
            if evidence(method):
                if method=='imugpt':
                    update_method(method,adapter_ready=True,runner='train.py',interpreter=str(HERE/'runtime_onellm_20261113/bin/python'),
                        execution_status='training_queue',reason='Genuine virtual corpus and all four train/checkpoint/reload smokes passed; admitted to the five-seed supervised queue.')
                else:
                    update_method(method,standalone_runner_ready=True,execution_status='native_validation_passed',
                        reason='Native pipeline validated; independent large-model/zero-shot evaluation queue on GPU1. Results are not copied into supervised ranking before protocol review.')
        todo=[]
        for task in tasks():
            key,method,file,interpreter,args=task;previous=journal['jobs'].get(key,{})
            if previous.get('status')=='failed' and previous.get('execution_fingerprint')==fingerprint(method,file):continue
            todo.append(task)
        allowed,reason=capacity();journal.update(updated_at=now(),waiting_reason=reason,pending_jobs=len(todo))
        if not todo or not allowed:
            write(STATE,journal);time.sleep(30);continue
        key,method,file,interpreter,args=todo[0]
        log=HERE/'logs'/f'{key}_20261113.log'
        code=HERE/'native_execution_source_20261113'/key;code.mkdir(parents=True,exist_ok=True)
        for name in [file,'api_adapters.py']:
            shutil.copyfile(HERE/name,code/name)
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DOWNLOAD_TIMEOUT='120')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        with log.open('ab') as output:
            child=subprocess.Popen([str(interpreter),'-u',str(HERE/file),*args],cwd=HERE,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            info=dict(method=method,status='running',pid=child.pid,started_at=now(),log=str(log),args=args,adapter_sha256=sha(HERE/file),execution_fingerprint=fingerprint(method,file))
            journal['jobs'][key]=info
            update_method(method,execution_status='native_running',native_job=key,native_pid=child.pid,
                          reason=f'GPU1: {key}. Full native weights/algorithm; no paid API calls. Supervised ranking unchanged until complete validation.')
            while child.poll() is None:
                journal.update(active_job=key,updated_at=now());write(STATE,journal);time.sleep(30)
            info.update(status='complete' if child.returncode==0 else 'failed',exit_code=child.returncode,completed_at=now())
            journal.pop('active_job',None);write(STATE,journal)
            update_method(method,execution_status='native_task_complete' if child.returncode==0 else 'native_validation_failed',
                native_pid=None,reason=f'{key}: {info["status"]}. '+('Next native step queued.' if child.returncode==0 else f'Inspect {log}; no result is fabricated.'))
            print(key,info['status'],flush=True)


if __name__=='__main__':main()
