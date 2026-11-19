"""Persistent, resumable oneHAR queue; never terminates other GPU jobs."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 2021, 2024, 3407]
DATASETS = ['uci_har', 'pamap2', 'usc_had', 'realworld_har']


def now(): return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def live(job):
    try:
        root = Path('/proc')/str(job['pid'])
        cmd = (root/'cmdline').read_bytes().replace(b'\0', b' ')
        return str(HERE/'run.py').encode() in cmd and (root/'stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def result(job):
    return HERE/'runs/onehar'/job['dataset']/f"seed_{job['seed']}"/job['fold']/'result.json'


def resources():
    disk = shutil.disk_usage(HERE).free/2**30
    ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:')))/2**20
    query = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=15)
    gpu = int(query.stdout.strip()) if query.returncode == 0 else 0
    return {'disk_free_GiB': round(disk, 2), 'available_RAM_GiB': round(ram, 2), 'GPU1_free_MiB': gpu,
            'can_launch': disk >= 34 and ram >= 32 and gpu >= 10000}


def jobs():
    rows = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for file in sorted((HERE/'manifests/splits'/dataset/f'seed_{seed}').glob('*.npz')):
                rows.append({'id': f'{dataset}__{seed}__{file.stem}', 'dataset': dataset, 'seed': seed, 'fold': file.stem})
    assert len(rows) == 190
    return rows


def main():
    with (HERE/'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        all_jobs = jobs()
        failures = read(HERE/'failures.json', {})
        active = read(HERE/'active.json', None)
        children = {}
        while True:
            if active and not live(active):
                child = children.pop(active['pid'], None)
                code = child.wait() if child else None
                if not result(active).exists():
                    failures[active['id']] = {'attempts': failures.get(active['id'], {}).get('attempts', 0)+1,
                                              'at': now(), 'returncode': code, 'log': active['log']}
                    write(HERE/'failures.json', failures)
                active = None
                write(HERE/'active.json', None)
            complete = [job for job in all_jobs if result(job).exists()]
            pending = [job for job in all_jobs if not result(job).exists()
                       and failures.get(job['id'], {}).get('attempts', 0) < 2]
            resource = resources()
            allowed = read(HERE/'preflight.json', {}).get('production_ready', False)
            if not active and pending and resource['can_launch'] and allowed:
                job = pending[0]
                log = HERE/'logs'/(job['id']+'.log')
                command = [sys.executable, '-u', str(HERE/'run.py'), '--dataset', job['dataset'],
                           '--seed', str(job['seed']), '--fold', job['fold'], '--gpu', '1']
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
                with log.open('a') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                             env=env, cwd=HERE, start_new_session=True)
                children[child.pid] = child
                active = {**job, 'pid': child.pid, 'started_at': now(), 'gpu': 1, 'log': str(log)}
                write(HERE/'active.json', active)
                print('START', active, flush=True)
            counts = Counter(job['dataset'] for job in complete)
            state = {'updated_at': now(), 'controller_pid': os.getpid(), 'method': 'onehar',
                     'variant': 'CDNet supervised adapter; no LLM synthesis', 'total_jobs': len(all_jobs),
                     'complete_jobs': len(complete), 'active': active,
                     'datasets': {d: {'completed_folds': counts[d], 'required_folds': sum(j['dataset']==d for j in all_jobs)}
                                  for d in DATASETS}, 'failures': failures, 'resources': resource,
                     'production_ready': allowed, 'paid_api_calls': 0,
                     'status': 'complete' if len(complete)==190 else 'running' if active else 'waiting'}
            write(HERE/'report.json', state)
            if len(complete) == len(all_jobs):
                break
            time.sleep(20)


if __name__ == '__main__': main()
