"""Persistent bounded queue for the requested HAR continuation on server 21."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'imu_reproduction_20261028'
SEEDS=[42,123,2021,2024,3407]
FOLDS={'uci_har':1,'pamap2':8,'realworld_har':15,'usc_had':14}
DATASETS=['usc_had','realworld_har','uci_har','pamap2']
METHODS=['deepconvlstm','if_convtransformer','unimts','limu_bert','hart','mobilehart',
         'ispl_inception','calf','gpt4ts','hargpt','imu2clip','imugpt','onellm','sensorllm','timellm','units','bymyeyes']


def now():return datetime.now(timezone.utc).isoformat()


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def alive(info):
    p=Path(f"/proc/{info['pid']}")
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ')
        return any(str(HERE/name).encode() in cmd for name in ['train.py','train_tf.py']) and (p/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def result_path(info):
    return HERE/'runs'/info['method']/info['dataset']/f"seed_{info['seed']}"/info['fold']/'result.json'


def capacity(config):
    if shutil.disk_usage(HERE).free < (config['disk_reserve_GiB']+2)*2**30:return False,'disk reserve'
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available < config['RAM_reserve_GiB']*2**30:return False,'RAM reserve'
    q=subprocess.run(['nvidia-smi',f"--id={config['gpu']}",'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if q.returncode:return False,'GPU query failed'
    if int(q.stdout.strip())<12000:return False,'waiting for 12 GB free on selected GPU'
    return True,None


def summarize(running,failures,config):
    original=read(OLD/'report.json',{})
    existing={(c['method'],c['dataset']):c for c in original.get('cells',[]) if c['status']=='complete'}
    methods=read(HERE/'readiness.json',{})
    cells=[]
    for method in METHODS:
        for dataset in FOLDS:
            key=(method,dataset)
            if key in existing:
                old=existing[key]
                cells.append(dict(method=method,dataset=dataset,status='complete_existing_raw_v2',
                    completed_folds=old['completed_folds'],required_folds=old['required_folds'],metrics=old['metrics'],
                    provenance=str(OLD/'report.json'),implementation_status='previously_run_calibration_status_retained'))
                continue
            out=HERE/'runs'/method/dataset
            results=[]
            for file in sorted(out.glob('seed_*/*/result.json')):
                row=read(file)
                if row.get('status')!='complete':continue
                assert row['method']==method and row['dataset']==dataset
                assert all(math.isfinite(v) and 0<=v<=1 for v in row['test'].values())
                results.append(dict(seed=row['seed'],fold=row['fold'],test=row['test'],
                    result_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),artifact_path=str(file),
                    recipe=row['recipe'],runtime_seconds=row['runtime_seconds'],
                    dataset_manifest_sha256=row['dataset_manifest_sha256']))
            required=5*FOLDS[dataset]
            assert len({(r['seed'],r['fold']) for r in results})==len(results)<=required
            jobs=[x['job'] for x in running.values() if x['method']==method and x['dataset']==dataset]
            errors=[x for x in failures.values() if x['method']==method and x['dataset']==dataset]
            ready=bool(methods.get(method,{}).get('adapter_ready'))
            data_ready=(HERE/'data'/dataset/'manifest.json').exists()
            status='running' if jobs else 'failed' if errors else 'queued' if ready and data_ready else 'data_pending' if ready else 'implementation_pending'
            c=dict(method=method,dataset=dataset,status=status,completed_folds=len(results),required_folds=required,
                running_jobs=jobs,errors=errors,source=methods.get(method,{}),fold_results=results)
            if len(results)==required:
                per_seed=[]
                for seed in SEEDS:
                    rr=[r for r in results if r['seed']==seed];assert len(rr)==FOLDS[dataset]
                    per_seed.append({k:sum(r['test'][k] for r in rr)/len(rr) for k in rr[0]['test']})
                metrics={}
                for key in per_seed[0]:
                    values=[r[key] for r in per_seed];avg=sum(values)/5
                    metrics[key]=avg;metrics[key+'_std']=(sum((v-avg)**2 for v in values)/5)**.5
                c.update(status='complete_pending_calibration',metrics=metrics,seed_metrics=per_seed,
                    protocol_status='canonical_candidate',source_type='measured_continuation_20261106')
            cells.append(c)
    special=[]
    retry=read(HERE/'state/robust_retry.json')
    if retry:
        file=result_path(retry)
        special.append(dict(**retry,status='complete' if file.exists() else 'running' if alive(retry) else 'failed',result=str(file) if file.exists() else None))
    report=dict(version='imu-continuation-20261106-v1',updated_at=now(),compute_server='10.130.151.21',web_server='10.130.151.16',
        controller_pid=os.getpid(),scope=dict(methods=len(METHODS),dataset_families=4,cells=len(cells)),
        excluded_by_user=['mosaic','onehar'],seeds=SEEDS,folds=FOLDS,
        protocol='Five seeds; official UCI split; all subject-held-out folds for PAMAP2 (12 classes), RealWorld (8 classes), USC-HAD (12 classes); next sorted subject held out for validation.',
        aggregation='Equal weight across test subjects within seed; mean/population SD across five seed means.',
        note='Only all-seed/all-fold complete results receive metrics. A completed candidate is not an independently calibrated exact paper reproduction. Prior results are retained.',
        data_preparation=read(HERE/'state/data_worker.json',{}),
        previous_repair=read(HERE/'state/previous_repair.json',{}),
        status_counts=dict(Counter(c['status'] for c in cells)),active_jobs=list(running.values()),
        queue=config,cells=cells,special_jobs=special)
    write(HERE/'report.json',report);write(HERE/'state/running.json',running)
    return report


def main():
    lock=(HERE/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=read(HERE/'queue.json')
    assert config['gpu']==0 and config['workers']<=3
    write(HERE/'state/controller.json',dict(pid=os.getpid(),started_at=now(),gpu=0,excluded_by_user=['mosaic','onehar']))
    running=read(HERE/'state/running.json',{})
    retry=read(HERE/'state/robust_retry.json')
    if retry and alive(retry):
        retry.setdefault('job',f"{retry['method']}__{retry['dataset']}__{retry['seed']}__{retry['fold']}")
        running[retry['job']]=retry
    failures=read(HERE/'state/failures.json',{})
    children={}
    while True:
        config=read(HERE/'queue.json')
        assert config['gpu']==0 and config['workers']<=3
        for key,info in list(running.items()):
            handle=children.get(key)
            active=handle[0].poll() is None if handle else alive(info)
            if active:continue
            file=result_path(info)
            if not file.exists():
                code=handle[0].returncode if handle else 'adopted_process_ended_without_result'
                failures[key]=dict(**info,exit_code=code,finished_at=now())
                write(HERE/'state/failures.json',failures)
            if handle:handle[1].close();children.pop(key)
            running.pop(key);print('finished',key,'complete' if file.exists() else 'failed',flush=True)
        report=summarize(running,failures,config)
        ready=read(HERE/'readiness.json',{})
        eligible={(c['method'],c['dataset']) for c in report['cells'] if c['status'] not in ['complete_existing_raw_v2','complete_pending_calibration']}
        if len(running)<config['workers'] and not (HERE/'STOP').exists():
            allowed,reason=capacity(config)
            if not allowed:write(HERE/'state/waiting.json',dict(updated_at=now(),reason=reason))
            else:
                for dataset in DATASETS:
                    if not (HERE/'data'/dataset/'manifest.json').exists():continue
                    splits=HERE/'manifests/splits'/dataset
                    for seed in SEEDS:
                        for split in sorted((splits/f'seed_{seed}').glob('*.npz')):
                            for method in METHODS:
                                if (method,dataset) not in eligible or not ready.get(method,{}).get('adapter_ready'):continue
                                key=f'{method}__{dataset}__{seed}__{split.stem}'
                                info=dict(job=key,method=method,dataset=dataset,seed=seed,fold=split.stem)
                                if key in running or key in failures or result_path(info).exists():continue
                                if len(running)>=config['workers']:break
                                settings=ready[method]
                                interpreter=settings.get('interpreter',sys.executable)
                                runner=settings.get('runner','train.py')
                                assert runner in ['train.py','train_tf.py']
                                args=[interpreter,'-u',str(HERE/runner),'--method',method,'--dataset',dataset,
                                    '--seed',str(seed),'--fold',split.stem,'--gpu',str(config['gpu']),
                                    '--memory-fraction',str(config['memory_fraction'])]
                                env=dict(os.environ,IOTBENCH_MATH_SDPA='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                                if runner=='train_tf.py':
                                    libs=[HERE/'tf_cuda_deps/nvidia/cudnn/lib']
                                    libs+=sorted(Path('/opt/conda/envs/fireredasr/lib/python3.10/site-packages/nvidia').glob('*/lib'))
                                    libs=[str(p) for p in libs if p.exists() and (p==HERE/'tf_cuda_deps/nvidia/cudnn/lib' or p.parent.name!='cudnn')]
                                    env['LD_LIBRARY_PATH']=':'.join(libs+[env.get('LD_LIBRARY_PATH','')])
                                log=(HERE/'logs'/f'{key}.log').open('ab')
                                process=subprocess.Popen(args,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                                info.update(pid=process.pid,started_at=now(),gpu=config['gpu'],log=str(Path(log.name)))
                                running[key]=info;children[key]=(process,log)
                                print('started',key,process.pid,flush=True)
                            if len(running)>=config['workers']:break
                        if len(running)>=config['workers']:break
                    if len(running)>=config['workers']:break
        summarize(running,failures,config)
        # Remain ready to admit newly prepared data/adapters without restarting
        # and without dropping already running work. STOP is an explicit pause.
        if (HERE/'STOP').exists():
            write(HERE/'state/waiting.json',dict(updated_at=now(),reason='STOP file; running jobs are allowed to finish'))
            if not running:break
        time.sleep(10)


if __name__=='__main__':main()
