"""Bounded GPU queue; only complete five-seed/all-fold combinations get a score."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from prepare import HERE, SEEDS, atomic_json, sha

DATASETS=['uci_har','wisdm','opportunity_a','hhar','pamap2']
METHODS=['mosaic','onehar','robusthar','unimts','if_convtransformer','tinyhar',
         'limu_bert','attend_and_discriminate','deepsense','deepconvlstm']
FOLDS={'uci_har':1,'wisdm':1,'opportunity_a':4,'hhar':9,'pamap2':8}


def report(running,failures):
    sources=json.loads((HERE/'methods.json').read_text())
    cells=[]
    for method in METHODS:
        for dataset in DATASETS:
            results=[]
            for p in sorted((HERE/'runs'/method/dataset).glob('seed_*/*/result.json')):
                row=json.loads(p.read_text())
                results.append({'seed':row['seed'],'fold':row['fold'],'test':row['test'],
                                'epochs_run':row['epochs_run'],'runtime_seconds':row['runtime_seconds'],
                                'result_sha256':sha(p),'artifact_path':str(p.relative_to(HERE))})
            expected=len(SEEDS)*FOLDS[dataset]
            jobs=[v for v in running.values() if v['method']==method and v['dataset']==dataset]
            errors=[v for v in failures.values() if v['method']==method and v['dataset']==dataset]
            source=sources[method]
            status='running' if jobs else ('failed' if errors else ('queued' if source.get('adapter_ready') else source['status']))
            cell={'method':method,'dataset':dataset,'status':status,'completed_folds':len(results),
                  'required_folds':expected,'seeds':SEEDS,'running_folds':[x['job'] for x in jobs],
                  'errors':errors,'fold_results':results,'source':source}
            if len(results)==expected:
                seeds=[]
                for seed in SEEDS:
                    group=[r for r in results if r['seed']==seed]
                    assert len(group)==FOLDS[dataset]
                    seeds.append({k:float(np.mean([r['test'][k] for r in group])) for k in group[0]['test']})
                scores={}
                for k in seeds[0]:
                    v=[r[k] for r in seeds]
                    scores[k]=float(np.mean(v));scores[k+'_std']=float(np.std(v))
                cell.update(status='complete',metrics=scores,seed_metrics=seeds,
                            validation_status='pending_paper_calibration_audit',
                            aggregation='unweighted mean across subjects within seed; mean/std across 5 seeds')
            cells.append(cell)
    result={'version':'20261028-imu-raw-v2','updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'scope':{'methods':10,'datasets':5,'combinations':50},
            'protocol':'Standard raw v2; 5 seeds; official UCI split / WISDM fixed subjects / all LOSO folds',
            'note':'Scores appear only after all required folds complete. Standard eligibility is audited separately.',
            'controller_pid':os.getpid(),'cells':cells}
    atomic_json(HERE/'report.json',result)
    return sources


def main():
    lock=(HERE/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(HERE/'controller.json',{'pid':os.getpid(),'started':time.time(),'gpu':3,'workers':2})
    processes={};running={};failures={}
    fp=HERE/'failures.json'
    if fp.exists():failures=json.loads(fp.read_text())
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    while True:
        for job,(p,log) in list(processes.items()):
            code=p.poll()
            if code is None:continue
            log.close();info=running.pop(job);processes.pop(job)
            if code:
                failures[job]={**info,'exit_code':code,'log':str(Path(log.name).relative_to(HERE))}
                atomic_json(fp,failures)
            print('finished',job,code,flush=True)
        sources=report(running,failures)
        for dataset in DATASETS:
            for method in METHODS:
                if not sources[method].get('adapter_ready'):continue
                for seed in SEEDS:
                    for split in sorted((HERE/'manifests/splits'/dataset/f'seed_{seed}').glob('*.npz')):
                        fold=split.stem;job=f'{method}__{dataset}__{seed}__{fold}'
                        out=HERE/'runs'/method/dataset/f'seed_{seed}'/fold
                        if (out/'result.json').exists() or job in running or job in failures:continue
                        if len(processes)>=2:break
                        args=[sys.executable,'-u',str(HERE/'train.py'),'--method',method,'--dataset',dataset,
                              '--seed',str(seed),'--fold',fold,'--gpu','3']
                        # Bound attention activation memory on the 113-channel track.
                        if dataset=='opportunity_a':args+=['--batch-size','32']
                        log=(HERE/'logs'/f'{job}.log').open('ab')
                        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                        processes[job]=(p,log)
                        running[job]={'method':method,'dataset':dataset,'seed':seed,'fold':fold,'job':job,'pid':p.pid}
                        print('started',job,p.pid,flush=True)
        report(running,failures)
        # Exit once every currently supported job finishes, including explicitly recorded failures.
        pending=any(c['status']=='queued' for c in json.loads((HERE/'report.json').read_text())['cells'])
        if not processes and not pending:
            print('queue drained; source/adapter blockers remain visible in report.json',flush=True)
            return
        time.sleep(5)


if __name__=='__main__':main()
