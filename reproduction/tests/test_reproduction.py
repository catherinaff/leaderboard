"""Dependency-free integrity and reproducibility checks; no model scores."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
SEEDS=[42,123,2021,2024,3407]
DATASETS={'uci_har':(6,1),'pamap2':(12,8),'usc_had':(12,14),'realworld_har':(8,15)}


def read(name):return json.loads((ROOT/name).read_text())


class ReproductionIntegrity(unittest.TestCase):
    def test_method_scope_and_honest_unrun_status(self):
        models=read('configs/methods.json')
        self.assertEqual(len(models),23)
        self.assertEqual(sum(m['scope']=='current' for m in models.values()),19)
        for name,m in models.items():
            self.assertTrue((ROOT/'methods'/name/'README.md').is_file())
            if m['entrypoint']:self.assertTrue((ROOT/m['entrypoint']).is_file(),name)
        self.assertEqual(models['mosaic']['implementation_class'],'paper_only')
        self.assertFalse(models['mosaic']['entrypoint'])
        for name in ['hargpt','bymyeyes']:self.assertEqual(models[name]['implementation_class'],'interface_only')
        self.assertEqual(models['onehar']['implementation_class'],'supervised_adapter')
        self.assertEqual(models['timellm']['implementation_class'],'task_extension')

    def test_fixed_source_versions_and_independent_comparison(self):
        pins=read('configs/upstream_sources.json')['sources']
        self.assertGreaterEqual(len(pins),19)
        for item in pins:
            self.assertRegex(item['metadata']['commit'],r'^[a-f0-9]{7,40}$')
            self.assertTrue(item['metadata']['repo'].startswith('https://github.com/'))
            self.assertFalse(Path(item['destination']).is_absolute())
            self.assertNotIn('..',Path(item['destination']).parts)
        check=read('evidence/official_source_comparison.json')
        self.assertEqual(check['matched'],25)
        self.assertEqual(check['mismatched'],0)
        self.assertEqual(check['unavailable'],0)
        for item in check['sources']:self.assertEqual(item['sha256'],item['fetched_sha256'])

    def test_executed_source_objects_are_not_silently_modified(self):
        external=set(read('evidence/external_source_objects.json')['not_revendored_sha256'])
        objects={p.stem:p for p in (ROOT/'evidence/executed_source_objects').iterdir() if p.is_file()}
        self.assertGreater(len(objects),100)
        for digest,file in objects.items():self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),digest,file.name)
        for bundle in read('evidence/executed_source_bundles.json').values():
            for digest in bundle.values():self.assertTrue(digest in objects or digest in external,digest)

    def test_native_labels_and_full_subject_seed_coverage(self):
        data=json.loads((REPO/'site/data/measured_results_20261119.json').read_text())
        seen=set()
        for r in data['records']:
            key=(r['method'],r['dataset']);self.assertNotIn(key,seen);seen.add(key)
            dataset='realworld_har' if r['family']=='realworld' else r['family']
            classes,subjects=DATASETS[dataset]
            self.assertEqual(r['num_classes'],classes)
            self.assertEqual(r['complete_subjects_per_seed'],subjects)
            self.assertEqual(r['completed_folds'],subjects*len(r['seeds']))
            self.assertTrue(set(r['seeds'])<=set(SEEDS))
            self.assertEqual(len(r['seeds']),len(set(r['seeds'])))
            if r['run_kind']=='zero_shot':
                self.assertEqual(r['method'],'onellm');self.assertEqual(r['seeds'],[42])
                self.assertNotIn('accuracy_std',r['metrics'])
            elif r['status']=='complete':self.assertEqual(set(r['seeds']),set(SEEDS))
            else:self.assertEqual(r['status'],'provisional_complete_seeds');self.assertLess(len(r['seeds']),5)
        self.assertEqual(sum(r['completed_folds'] for r in data['records']),data['totals']['verified_fold_records'])

    def test_recorded_settings_distinguish_caps_from_actual_epochs(self):
        recipes=read('configs/training_recipes.json')
        self.assertEqual(recipes['seeds'],SEEDS)
        for cell in recipes['cells']:
            if cell['run_kind']=='zero_shot':continue
            self.assertTrue(cell['recipe_variants'])
            for r in cell['recipe_variants']:
                self.assertIn('optimizer',r);self.assertGreater(r['lr'],0);self.assertGreater(r['batch_size'],0)
                if 'max_epochs' in r and cell['actual_epochs']:
                    self.assertLessEqual(cell['actual_epochs']['max'],r['max_epochs'])
        for name,caps in [('onehar',{80}),('sensorllm',{30})]:
            actual={r['max_epochs'] for c in recipes['cells'] if c['method']==name for r in c['recipe_variants']}
            self.assertEqual(actual,caps)

    def test_disabled_api_configuration_has_no_credentials(self):
        for row in read('configs/api.disabled.example.json').values():
            self.assertIs(row['allow_paid_calls'],False)
            self.assertEqual(row['max_calls'],0);self.assertEqual(row['budget_usd'],0)
            self.assertNotIn('api_key',row)
        runtime=read('evidence/runtime_review.json')
        self.assertEqual(runtime['status'],'passed');self.assertEqual(runtime['paid_api_calls'],0)
        self.assertTrue(all(c['passed'] for c in runtime['checks']))

    def test_portable_launch_plans_do_not_need_ml_assets(self):
        script=ROOT/'scripts/run_benchmark.py'
        def plan(method,dataset='uci_har',*extra):
            return json.loads(subprocess.check_output([sys.executable,str(script),'--method',method,'--dataset',dataset,'--dry-run',*extra],text=True))
        rows=plan('deepconvlstm','uci_har','--seeds','42','--fold','official');self.assertEqual(len(rows),1)
        self.assertEqual(len(plan('deepconvlstm','pamap2')),40)
        self.assertEqual(len(plan('onehar','realworld_har')),75)
        self.assertEqual(len(plan('onellm')),1)
        self.assertIn('--dry-run-matrix',plan('hargpt')[0])
        for cmd in plan('bymyeyes'):self.assertNotIn('--execute',cmd)

    def test_authored_current_code_parses_without_import_side_effects(self):
        for folder in [ROOT/'scripts',ROOT/'experiments/imu_continuation_20261106',ROOT/'experiments/imu_reproduction_20261028',ROOT/'experiments/onehar_canonical_20261119']:
            for file in folder.glob('*.py'):
                with self.subTest(file=file.name):ast.parse(file.read_text(),filename=str(file))

    def test_cached_asset_still_creates_required_reference_alias(self):
        file=ROOT/'scripts/fetch_assets.py'
        spec=importlib.util.spec_from_file_location('asset_fetch_fixture',file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='asset-pin-fixture-') as name:
            directory=Path(name);(directory/'asset').write_bytes(b'local fixture')
            item={'destination':'asset','sha256':hashlib.sha256(b'local fixture').hexdigest(),'url':'https://example.invalid/unreachable','aliases':['refs/alias']}
            with patch.object(module,'ROOT',directory),patch.object(module.urllib.request,'urlopen',side_effect=AssertionError('Cached valid bytes must not download')):
                module.checked_url(item)
            self.assertTrue((directory/'refs/alias').is_symlink())
            self.assertEqual((directory/'refs/alias').read_bytes(),b'local fixture')

    def test_method_readme_local_links_exist(self):
        files=[ROOT/'README.md',REPO/'README.md',REPO/'README.zh-CN.md',*list((ROOT/'methods').glob('*/README.md'))]
        for file in files:
            for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
                if target.startswith(('http:','https:','#')):continue
                path=target.split('#',1)[0]
                if path:self.assertTrue((file.parent/path).exists(),f'{file}: {target}')


if __name__=='__main__':unittest.main()
