"""Bounded oneHAR UCI/PAMAP result updates; no incomplete subject subsets."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import math
import statistics

VERSION = 'onehar-canonical-leaderboard-20261119-v1'
SEEDS = [42, 123, 2021, 2024, 3407]
TARGETS = {'uci_har': (1, 6), 'pamap2': (8, 12)}
METRICS = ['accuracy', 'f1_macro', 'f1_weighted']


def validate_patch(patch):
    assert patch['version'] == VERSION
    assert patch['paid_api_calls'] == 0
    records = patch['records']
    assert 1 <= len(records) <= 2
    assert len({r['dataset'] for r in records}) == len(records)
    for r in records:
        assert r['method'] == 'onehar' and r['dataset'] in TARGETS
        n, classes = TARGETS[r['dataset']]
        assert r['family'] == r['dataset'] and r['num_classes'] == classes
        assert r['run_kind'] == 'supervised' and r['verified'] is True
        seeds = r['seeds']
        assert seeds and len(set(seeds)) == len(seeds) and set(seeds) <= set(SEEDS)
        assert r['required_seeds'] == SEEDS
        assert r['complete_subjects_per_seed'] == n
        assert r['completed_folds'] == len(seeds)*n and r['required_folds'] == 5*n
        assert r['status'] == ('complete' if len(seeds) == 5 else 'provisional_complete_seeds')
        rows = r['fold_results']
        assert len(rows) == len(seeds)*n
        assert len({(row['seed'], row['fold']) for row in rows}) == len(rows)
        expected = {'official'} if n == 1 else {f'test_{s}_val_{101+(s-100)%8}' for s in range(101, 109)}
        for seed in seeds:
            assert {v['fold'] for v in rows if v['seed'] == seed} == expected
        assert set(v['seed'] for v in rows) == set(seeds)
        for row in rows:
            assert len(row['result_sha256']) == 64 and int(row['result_sha256'], 16) >= 0
            for metric in METRICS:
                value = row['test'][metric]
                assert isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
        assert r['variant_label'] and r['calibration']
        for metric in METRICS:
            means = [statistics.mean(v['test'][metric] for v in rows if v['seed'] == seed) for seed in seeds]
            assert abs(statistics.mean(means)-r['metrics'][metric]) < 1e-10
            if len(seeds) > 1:
                assert abs(statistics.pstdev(means)-r['metrics'][metric+'_std']) < 1e-10
            else:
                assert metric+'_std' not in r['metrics']
    return records


def merge(base, patch):
    records = validate_patch(patch)
    value = copy.deepcopy(base)
    current = {(r['method'], r['dataset']): r for r in value['records']}
    changed = False
    for r in records:
        key = (r['method'], r['dataset'])
        old = current.get(key)
        if old:
            assert set(old['seeds']) <= set(r['seeds']), 'Reject loss of published complete seed coverage'
            if old['fold_results'] == r['fold_results'] and old['metrics'] == r['metrics']:
                continue
        current[key] = r
        changed = True
    if not changed:
        return base, False
    value['records'] = sorted(current.values(), key=lambda r: (r['method'], r['dataset']))
    value.setdefault('base_snapshot_at', base['snapshot_at'])
    value['updated_at'] = value['snapshot_at'] = patch['updated_at']
    value['incremental_onehar'] = {'updated_at': patch['updated_at'], 'datasets': list(TARGETS),
                                  'policy': 'Complete all-subject seeds only; source and adapter notes retained.'}
    rows = value['records']
    value['totals'] = {'records': len(rows), 'methods': len({r['method'] for r in rows}),
                       'complete_supervised': sum(r['status'] == 'complete' for r in rows),
                       'provisional': sum(r['status'] == 'provisional_complete_seeds' for r in rows),
                       'zero_shot': sum(r['run_kind'] == 'zero_shot' for r in rows),
                       'verified_fold_records': sum(r['completed_folds'] for r in rows)}
    return value, True


def receive_patch(raw, release):
    """Called only inside the existing authenticated, locked SSH receiver."""
    patch = json.loads(raw)
    relative = 'public/data/measured_results_20261119.json'
    path = release/relative
    base = json.loads(path.read_text())
    value, changed = merge(base, patch)
    if changed:
        body = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        temporary = path.with_name(path.name+'.onehar-tmp')
        temporary.write_bytes(body); temporary.chmod(0o644); temporary.replace(path)
        manifest_path = release/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][relative] = hashlib.sha256(body).hexdigest()
        temporary = manifest_path.with_name('manifest.onehar-tmp.json')
        temporary.write_text(json.dumps(manifest, indent=2)+'\n'); temporary.replace(manifest_path)
    print(json.dumps({'status': 'published' if changed else 'unchanged',
                      'totals': value['totals'], 'updated_at': value['updated_at']}))
