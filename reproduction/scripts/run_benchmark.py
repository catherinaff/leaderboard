"""Portable per-fold launcher for the reviewed, unchanged training adapters."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'experiments/imu_continuation_20261106'
OLD=ROOT/'experiments/imu_reproduction_20261028'
ONE=ROOT/'experiments/onehar_canonical_20261119'
SEEDS=[42,123,2021,2024,3407]
DATASETS={'uci_har':(1,1),'pamap2':(8,101),'usc_had':(14,1),'realworld_har':(15,1)}


def fold_names(dataset):
    n,first=DATASETS[dataset]
    return ['official'] if n==1 else [f'test_{s}_val_{first+(s-first+1)%n}' for s in range(first,first+n)]


def commands(a):
    tf={'hart','mobilehart','ispl_inception'}
    legacy={'robusthar','tinyhar','deepsense','attend_and_discriminate'}
    metadata=json.loads((ROOT/'configs/methods.json').read_text())
    if a.method not in metadata:raise ValueError('Unknown method')
    if a.method=='mosaic':raise ValueError('MOSAIC is paper-only; no implemented runner is claimed')
    if a.method in {'hargpt','bymyeyes'}:
        return [[a.python,str(NEW/'api_adapters.py'),'--dry-run-matrix']]
    if a.method=='onellm':
        cmd=[a.python,str(NEW/'onellm_adapter.py'),'--dataset',a.dataset,'--gpu',str(a.gpu)]
        if a.fold!='all':cmd+=['--fold',a.fold]
        return [cmd]  # Frozen model: do not fabricate five independent fits.
    folds=fold_names(a.dataset) if a.fold=='all' else [a.fold]
    if any(f not in fold_names(a.dataset) for f in folds):raise ValueError('Fold is not part of the frozen canonical subject set')
    rows=[]
    for seed in a.seeds:
        for fold in folds:
            if a.method=='onehar':script=ONE/'run.py';extra=[]
            elif a.method=='sensorllm':script=NEW/'sensorllm_adapter.py';extra=['--epochs','30']
            elif a.method in tf:script=NEW/'train_tf.py';extra=['--method',a.method]
            elif a.method in legacy:
                if a.dataset not in {'uci_har','pamap2'}:raise ValueError('Historical runner needs its original five-dataset preparation; use original CLI for other legacy datasets')
                script=OLD/'train.py';extra=['--method',a.method]
            else:script=NEW/'train.py';extra=['--method',a.method]
            rows.append([a.python,str(script),*extra,'--dataset',a.dataset,'--seed',str(seed),'--fold',fold,'--gpu',str(a.gpu)])
    return rows


def bridge(method):
    for name in ['data','manifests']:
        target=ONE/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(Path('../imu_continuation_20261106')/name,target_is_directory=True)
    if method=='onehar':
        release=ROOT/'experiments/onehar_20261028'
        if not (release/'runtime/test.py').exists():
            spec=importlib.util.spec_from_file_location('onehar_runtime_setup',release/'evaluate_official.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.setup_runtime()


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--dataset',choices=list(DATASETS),required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--fold',default='all');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--python',default=sys.executable);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not set(a.seeds)<=set(SEEDS):p.error('Use the documented five-seed set')
    plan=commands(a)
    if a.dry_run:print(json.dumps(plan,indent=2));return
    bridge(a.method)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for command in plan:subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
