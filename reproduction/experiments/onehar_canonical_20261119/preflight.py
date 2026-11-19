"""Verify the common frozen data/splits, then exercise all four real adapters."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent/'imu_continuation_20261106'
SEEDS = [42, 123, 2021, 2024, 3407]
FOLDS = {'uci_har': 1, 'pamap2': 8, 'usc_had': 14, 'realworld_har': 15}
CLASSES = {'uci_har': 6, 'pamap2': 12, 'usc_had': 12, 'realworld_har': 8}


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    dest = HERE/name
    temp = dest.with_name(dest.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(dest)


def main():
    for name in ['data', 'manifests']:
        path = HERE/name
        if not path.exists(): path.symlink_to(SHARED/name, target_is_directory=True)
        assert path.resolve() == (SHARED/name).resolve()
    (HERE/'logs').mkdir(exist_ok=True)
    proof = {}
    for dataset, subjects in FOLDS.items():
        root = HERE/'data'/dataset
        manifest = json.loads((root/'manifest.json').read_text())
        assert len(manifest['classes']) == CLASSES[dataset]
        y = np.load(root/'y.npy'); people = np.load(root/'subjects.npy')
        assert set(y) == set(range(CLASSES[dataset]))
        for name, digest in manifest['arrays_sha256'].items():
            assert sha(root/(name+'.npy')) == digest
        splits = {}
        for seed in SEEDS:
            files = sorted((HERE/'manifests/splits'/dataset/f'seed_{seed}').glob('*.npz'))
            assert len(files) == subjects
            for file in files:
                with np.load(file) as split:
                    tr, va, te = [split[k] for k in ['train', 'validation', 'test']]
                    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
                    assert not set(people[tr])&set(people[te])
                    if dataset != 'uci_har':
                        assert len(set(people[va])) == len(set(people[te])) == 1
                        assert not set(people[tr])&set(people[va])
                        assert not set(people[va])&set(people[te])
                splits[str(file.relative_to(HERE/'manifests/splits'))] = sha(file)
        proof[dataset] = {'num_classes': len(manifest['classes']), 'classes': manifest['classes'],
                          'manifest_sha256': sha(root/'manifest.json'), 'split_sha256': splits}
        print('DATA_VERIFIED', dataset, len(splits), 'splits', flush=True)
    save('protocol.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'method': 'onehar',
                          'variant': 'Author CDNet supervised adapter, without LLM synthesis / joint training',
                          'source': str(SHARED), 'shared_runner_sha256': sha(SHARED/'train.py'),
                          'seeds': SEEDS, 'folds': FOLDS, 'total_jobs': 190, 'datasets': proof,
                          'user_resumed_onehar': True, 'mosaic_remains_paused': True,
                          'gpu': 1, 'max_workers': 1, 'memory_fraction': .16,
                          'disk_min_free_GiB': 34, 'paid_api_calls': 0})
    checks = []
    for dataset in FOLDS:
        file = sorted((HERE/'manifests/splits'/dataset/'seed_42').glob('*.npz'))[0]
        log = HERE/'logs'/('smoke_'+dataset+'.log')
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', str(HERE/'run.py'), '--dataset', dataset,
                            '--seed', '42', '--fold', file.stem, '--gpu', '1', '--epochs', '2', '--smoke'],
                           cwd=HERE, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'))
        out = HERE/'smoke/onehar'/dataset/'seed_42'/file.stem
        r = json.loads((out/'result.json').read_text())
        assert r['status'] == 'smoke_complete' and r['method'] == 'onehar'
        with np.load(out/'predictions.npz') as pred:
            logits = pred['logits']; y = pred['y_true']; p = pred['y_pred']
            assert logits.shape == (len(y), CLASSES[dataset]) and np.isfinite(logits).all()
            assert np.array_equal(p, logits.argmax(1))
            assert abs(accuracy_score(y, p)-r['test']['accuracy']) < 1e-10
            assert abs(f1_score(y, p, labels=list(range(CLASSES[dataset])), average='macro', zero_division=0)-r['test']['f1_macro']) < 1e-10
        for name, digest in r['artifacts'].items(): assert sha(out/name) == digest
        checkpoint = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        assert not any(k.startswith('label_encoder.bert') for k in checkpoint['state_dict'])
        checks.append({'dataset': dataset, 'status': 'passed', 'num_classes': CLASSES[dataset],
                       'checkpoint_bytes': (out/'best.pt').stat().st_size,
                       'cache_audit': r['recipe']['cache_audit'], 'result_sha256': sha(out/'result.json')})
        save('preflight.json', {'production_ready': False, 'checks': checks})
        print('SMOKE_VERIFIED', dataset, checks[-1], flush=True)
    estimated = sum(c['checkpoint_bytes'] * 5 * FOLDS[c['dataset']] for c in checks) * 1.15
    free = shutil.disk_usage(HERE).free
    assert free - estimated > 32*2**30, (free, estimated)
    save('preflight.json', {'production_ready': True, 'checks': checks,
                           'estimated_artifact_budget_GiB': round(estimated/2**30, 2),
                           'disk_free_GiB': round(free/2**30, 2),
                           'passed_at': datetime.now(timezone.utc).isoformat(),
                           'no_smoke_scores_published': True, 'paid_api_calls': 0})
    print('PRODUCTION_READY', flush=True)


if __name__ == '__main__': main()
