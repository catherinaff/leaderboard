"""Audit completed UCI/PAMAP seeds and publish through the existing SSH key."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent/'imu_continuation_20261106'
sys.path.insert(0, str(SHARED))
from audit_results_20261111 import verify
sys.path.insert(0, str(HERE))
from publish_contract import VERSION, SEEDS, TARGETS, METRICS, validate_patch


def build_patch():
    records = []
    for dataset, (n, classes) in TARGETS.items():
        rows = []
        recipes = []
        for seed in SEEDS:
            expected = {f.stem for f in (HERE/'manifests/splits'/dataset/f'seed_{seed}').glob('*.npz')}
            assert len(expected) == n
            found = {f.parent.name: f for f in (HERE/'runs/onehar'/dataset/f'seed_{seed}').glob('*/result.json')}
            if set(found) != expected:
                continue
            for fold in sorted(expected):
                file = found[fold]
                audited = verify(file)
                r = json.loads(file.read_text()); recipes.append(r['recipe'])
                assert r['recipe']['implementation_type'] == 'author_architecture_with_disclosed_supervised_adapter'
                assert r['recipe']['cache_audit']['author_pool_axis_fix']
                rows.append({'seed': seed, 'fold': fold, 'test': r['test'],
                             'result_sha256': audited['result_sha256'], 'path': str(file)})
        if not rows:
            continue
        seeds = [seed for seed in SEEDS if any(r['seed'] == seed for r in rows)]
        seed_metrics = {str(seed): {metric: statistics.mean(r['test'][metric] for r in rows if r['seed'] == seed)
                                   for metric in METRICS} for seed in seeds}
        metrics = {metric: statistics.mean(v[metric] for v in seed_metrics.values()) for metric in METRICS}
        if len(seeds) > 1:
            metrics.update({metric+'_std': statistics.pstdev(v[metric] for v in seed_metrics.values()) for metric in METRICS})
        recipe = recipes[0]
        records.append({'id': f'measured-20261119:onehar:{dataset}', 'method': 'onehar', 'dataset': dataset,
                        'family': dataset, 'num_classes': classes, 'run_kind': 'supervised',
                        'status': 'complete' if len(seeds) == 5 else 'provisional_complete_seeds',
                        'primary_metric': 'accuracy', 'metrics': metrics, 'seed_metrics': seed_metrics,
                        'seeds': seeds, 'required_seeds': SEEDS, 'complete_subjects_per_seed': n,
                        'completed_folds': len(rows), 'required_folds': 5*n, 'verified': True,
                        'aggregation': 'Equal test-subject weight within every complete seed; equal seed weight; population SD across complete seeds.',
                        'protocol': 'Same canonical UCI official test / PAMAP all8-subject LOSO frozen splits and train-only median/z-score as the other methods.',
                        'calibration': recipe['calibration'], 'variant_label': recipe['variant_label'],
                        'recipe': recipe, 'fold_results': rows,
                        'dataset_manifest_sha256': r['dataset_manifest_sha256']})
    patch = {'version': VERSION, 'updated_at': datetime.now(timezone.utc).isoformat(),
             'records': records, 'paid_api_calls': 0}
    if records:
        validate_patch(patch)
    return patch


def publish(patch):
    raw = (json.dumps(patch, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
    command = ['ssh', '-T', '-i', str(SHARED/'report_ed25519'), '-o', 'IdentitiesOnly=yes',
               '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
               '-o', 'UserKnownHostsFile='+str(SHARED/'server16_known_hosts'), '-p', '46429', 'root@10.130.151.16']
    result = subprocess.run(command, input=raw, capture_output=True, timeout=60)
    status = {'updated_at': patch['updated_at'], 'pid': os.getpid(),
              'status': 'published' if result.returncode == 0 else 'error',
              'response': result.stdout.decode(errors='replace')[:2000],
              'error': result.stderr.decode(errors='replace')[:2000],
              'coverage': {r['dataset']: r['seeds'] for r in patch['records']}}
    dest = HERE/'publication_status.json'
    temp = dest.with_suffix('.tmp'); temp.write_text(json.dumps(status, indent=2)+'\n'); temp.replace(dest)
    if result.returncode:
        raise RuntimeError(status['error'])
    (HERE/'published_patch.json').write_bytes(raw)
    print(json.dumps(status), flush=True)


def main(once=False):
    with (HERE/'publication.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            try:
                patch = build_patch()
                digest = hashlib.sha256(json.dumps(patch['records'], sort_keys=True).encode()).hexdigest()
                if patch['records'] and digest != previous:
                    publish(patch); previous = digest
                if len(patch['records']) == 2 and all(len(r['seeds']) == 5 for r in patch['records']):
                    break
            except Exception as exc:
                if once:
                    raise
                print(type(exc).__name__, str(exc), flush=True)
            if once:
                break
            time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true')
    main(parser.parse_args().once)
