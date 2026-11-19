"""Forced-command SSH receiver: accepts only this experiment's bounded JSON report."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

METHODS={'mosaic','onehar','robusthar','unimts','if_convtransformer','tinyhar',
         'limu_bert','attend_and_discriminate','deepsense','deepconvlstm'}
FOLDS={'uci_har':1,'wisdm':1,'opportunity_a':4,'hhar':9,'pamap2':8}
SEEDS={42,123,2021,2024,3407}


def validate(raw):
    if len(raw)>8*1024*1024:raise ValueError('report too large')
    value=json.loads(raw)
    if value['version']!='20261028-imu-raw-v2':raise ValueError('wrong experiment')
    cells=value['cells']
    expected={(m,d) for m in METHODS for d in FOLDS}
    if len(cells)!=50 or {(c['method'],c['dataset']) for c in cells}!=expected:raise ValueError('wrong scope')
    for c in cells:
        required=5*FOLDS[c['dataset']]
        if c['required_folds']!=required or not 0<=c['completed_folds']<=required:raise ValueError('invalid fold counts')
        rows=c['fold_results']
        if len(rows)!=c['completed_folds'] or len({(r['seed'],r['fold']) for r in rows})!=len(rows):raise ValueError('duplicate/missing fold')
        for r in rows:
            if r['seed'] not in SEEDS:raise ValueError('invalid seed')
            for v in r['test'].values():
                if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('invalid metric')
        if c['status']=='complete':
            if len(rows)!=required:raise ValueError('premature score')
            for seed in SEEDS:
                if sum(r['seed']==seed for r in rows)!=FOLDS[c['dataset']]:raise ValueError('incomplete seed')
            for metric in ['accuracy','f1_macro','f1_weighted']:
                means=[sum(r['test'][metric] for r in rows if r['seed']==s)/FOLDS[c['dataset']] for s in sorted(SEEDS)]
                mean=sum(means)/5
                std=(sum((x-mean)**2 for x in means)/5)**.5
                if abs(mean-c['metrics'][metric])>1e-10 or abs(std-c['metrics'][metric+'_std'])>1e-10:raise ValueError('incorrect aggregate')
    return value


def write(path,data):
    tmp=path.with_name(path.name+'.imu-tmp')
    with tmp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    tmp.chmod(0o644);tmp.replace(path)


def promote(report,registry,policy):
    """Only independently approved implementations/data can enter Standard."""
    invalid={c['method']:set(c['source'].get('superseded_model_sha256',[])) for c in report['cells']}
    for method in registry:
        for r in method.get('results',[]):
            if r.get('run_suite')=='20261028-imu-raw-v2' and r.get('model_sha256') in invalid.get(method['id'],set()):
                r['reproduction_status']='superseded_implementation_variant'
                r['review_note']='Superseded after a source-equation audit. Preserved as a measured experimental variant; excluded from current reproduction and Standard.'
        if method['id'] in ['tinyhar','if_convtransformer','unimts']:
            for r in method.get('results',[]):
                if r.get('protocol_status')=='canonical' and r.get('source')!='official_code_standard_raw_v2':
                    r['reproduction_status']='legacy_protocol_review_not_standard_main'
                    r['review_note']='Historical adapter predates the raw-v2 audit. Five-seed/all-fold coverage, validation-only checkpointing, train-only scaling and sensor selection must be verified before Standard use. Retained as historical evidence.'
    approved=policy.get('methods',{})
    official={'tinyhar','attend_and_discriminate','if_convtransformer','unimts','limu_bert'}
    fresh_sources={'official_code_standard_raw_v2','paper_reimplementation_standard_raw_v2',
                   'official_code_adapter_raw_v2','paper_reimplementation_raw_v2'}
    for cell in report['cells']:
        if cell['status']!='complete':continue
        rules=approved.get(cell['method'])
        rows=cell['fold_results']
        valid=bool(rules) and all(r.get('model_sha256')==rules['model_sha256'] and
                  r.get('official_commit')==rules['official_commit'] and
                  r.get('dataset_manifest_sha256')==policy['datasets'][cell['dataset']]
                  for r in rows)
        cell['validation_status']='standard_eligible' if valid else 'pending_paper_calibration_audit'
        if valid:cell['calibration_evidence']=rules['calibrations']
        is_official=cell['method'] in official
        source=('official_code_standard_raw_v2' if is_official else 'paper_reimplementation_standard_raw_v2') if valid else ('official_code_adapter_raw_v2' if is_official else 'paper_reimplementation_raw_v2')
        primary='f1_weighted' if cell['dataset']=='opportunity_a' else 'accuracy'
        incoming={'benchmark_id':cell['dataset'],'primary_metric':primary,'primary_score':cell['metrics'][primary],
                  'metrics':cell['metrics'],'source':source,'protocol_status':'canonical' if valid else 'canonical_candidate',
                  'official_code_used':is_official,'paper_reimplementation':not is_official,'official_repo':cell['source']['repo'],
                  'official_commit':rows[0].get('official_commit'),'model_sha256':rows[0].get('model_sha256'),
                  'run_suite':'20261028-imu-raw-v2',
                  'standard_protocol_version':'2026-10-28-raw-v2','seeds':sorted(SEEDS),
                  'folds':FOLDS[cell['dataset']],'completed_fold_runs':len(rows),
                  'runtime_seconds':sum(r['runtime_seconds'] for r in rows),
                  'calibration_status':'standard_after_two_dataset_paper_calibration_passed' if valid else 'pending_paper_calibration',
                  'paper_calibration_gate':'saved_two_dataset_calibrations_reaudited' if valid else 'not_yet_passed',
                  'calibration_evidence':rules['calibrations'] if valid else [],'reproduction_status':'complete' if valid else 'complete_pending_calibration',
                  'dataset_manifest_sha256':rows[0].get('dataset_manifest_sha256'),
                  'seed_fold_results':rows,'run_manifest_url':'/data/imu_reproduction_20261028.json',
                  'protocol_note':'Raw-v2: official UCI split; WISDM 1-28/29-30/31-36; PAMAP2/HHAR/OPPORTUNITY full LOSO with a separate validation subject. Training-only median/scaling. Five seeds, validation primary-metric checkpoint selection; test predictions saved after training. Mean/std over seed-level subject-fold means.',
                  'review_note':rules['basis'] if valid else cell['source']['note']}
        method=next(m for m in registry if m['id']==cell['method'])
        old=next((i for i,r in enumerate(method['results']) if r.get('benchmark_id')==cell['dataset'] and
                  r.get('source') in fresh_sources and r.get('model_sha256')==incoming['model_sha256'] and
                  r.get('dataset_manifest_sha256')==incoming['dataset_manifest_sha256']),None)
        if old is None:method['results'].append(incoming)
        else:method['results'][old]=incoming
    report['calibration_policy']={'methods':list(approved),'basis':'Independent artifact audit; only matching model/data fingerprints are promoted.'}
    return report,registry


if __name__=='__main__':
    raw=sys.stdin.buffer.read(8*1024*1024+1)
    value=validate(raw)
    lock=Path('/var/lib/iotbench-leaderboard/imu_report.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    release=Path('/srv/iotbench-leaderboard/current').resolve(strict=True)
    if release.parent!=Path('/srv/iotbench-leaderboard/releases'):raise ValueError('invalid release path')
    relative='public/data/imu_reproduction_20261028.json'
    policy_path=Path('/opt/iotbench-leaderboard/imu_calibration_policy.json')
    if policy_path.exists():
        registry_path=release/'public/api/results'
        value,registry=promote(value,json.loads(registry_path.read_text()),json.loads(policy_path.read_text()))
        raw=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        registry_raw=(json.dumps(registry,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        if registry_raw!=registry_path.read_bytes():write(registry_path,registry_raw)
    write(release/relative,raw)
    manifest=json.loads((release/'manifest.json').read_text())
    manifest['files'][relative]=hashlib.sha256(raw).hexdigest()
    manifest['files']['public/api/results']=hashlib.sha256((release/'public/api/results').read_bytes()).hexdigest()
    write(release/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps({'status':'published','updated_at':value['updated_at'],'sha256':hashlib.sha256(raw).hexdigest()}))
