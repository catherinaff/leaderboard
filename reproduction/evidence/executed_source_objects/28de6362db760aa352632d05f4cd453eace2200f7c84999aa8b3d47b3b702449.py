"""Read-only receiver regressions using the captured real report and registry."""
import copy
import json
from pathlib import Path
from publish_report import promote, validate

HERE = Path(__file__).resolve().parent
report = validate((HERE/'preview_report.json').read_bytes())
registry = json.loads((HERE/'preview_results.json').read_text())
policy = json.loads((HERE/'calibration_policy.json').read_text())

first_report, first_registry = promote(copy.deepcopy(report), copy.deepcopy(registry), policy)
second_report, second_registry = promote(copy.deepcopy(first_report), copy.deepcopy(first_registry), policy)
assert first_registry == second_registry, 'Repeated publication must not duplicate results'
assert first_report == second_report
old = [r for m in first_registry if m['id']=='robusthar' for r in m['results']
       if r.get('run_suite')=='20261028-imu-raw-v2']
assert old and all(r['reproduction_status']=='superseded_implementation_variant' for r in old)

changed = copy.deepcopy(report)
tiny = next(c for c in changed['cells'] if c['method']=='tinyhar' and c['dataset']=='uci_har')
assert tiny['status']=='complete'
for row in tiny['fold_results']:
    row['model_sha256'] = '0' * 64
changed, changed_registry = promote(changed, copy.deepcopy(registry), policy)
assert tiny['validation_status']=='pending_paper_calibration_audit'
new = [r for m in changed_registry if m['id']=='tinyhar' for r in m['results']
       if r.get('run_suite')=='20261028-imu-raw-v2' and r.get('model_sha256')=='0'*64]
assert len(new)==1 and new[0]['protocol_status']=='canonical_candidate'
assert new[0]['source']=='official_code_adapter_raw_v2'
assert any(r['source']=='official_code_standard_raw_v2' and r['benchmark_id']=='uci_har'
           for m in changed_registry if m['id']=='tinyhar' for r in m['results'])

def rejected(mutator):
    bad = copy.deepcopy(report)
    cell = next(c for c in bad['cells'] if c['method']=='tinyhar' and c['dataset']=='uci_har')
    mutator(cell)
    try:
        validate(json.dumps(bad).encode())
    except (ValueError, AssertionError):
        return
    raise AssertionError('Invalid report accepted')

rejected(lambda c: c['fold_results'].pop())
rejected(lambda c: c['fold_results'].__setitem__(0, copy.deepcopy(c['fold_results'][1])))
rejected(lambda c: c['fold_results'][0]['test'].__setitem__('accuracy', float('nan')))
rejected(lambda c: c['metrics'].__setitem__('accuracy', .999))
print('PASS bounded report validation, idempotent publication, superseding, fingerprint gate and approved history preservation.')
