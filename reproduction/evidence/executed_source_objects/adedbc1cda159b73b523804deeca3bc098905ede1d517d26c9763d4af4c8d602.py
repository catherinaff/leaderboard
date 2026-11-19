"""Publish verified HAR results using a fixed, score-independent cutoff.

Only complete subject-fold sets contribute. All finished seeds are used; a
smoke, failed job or incomplete subject set cannot become a leaderboard value.
"""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score,f1_score

HERE=Path('/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106')
OLD=HERE.parent/'imu_reproduction_20261028'
OUT=HERE.parent/'leaderboard_refresh_20261119'
SEEDS=[42,123,2021,2024,3407]
DATASETS={'uci_har':(1,6),'realworld_har':(15,8),'pamap2':(8,12),'usc_had':(14,12)}
METRICS=['accuracy','f1_macro','f1_weighted']
os.environ.setdefault('OMP_NUM_THREADS','2')


def sha(file):
    h=hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(file):return json.loads(file.read_text())


def write(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    temp=file.with_name(file.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(file)


def aggregate(rows):
    by_seed={}
    for seed in sorted({r['seed'] for r in rows}):
        chosen=[r for r in rows if r['seed']==seed]
        by_seed[str(seed)]={key:float(np.mean([r['test'][key] for r in chosen])) for key in METRICS}
    values={key:float(np.mean([v[key] for v in by_seed.values()])) for key in METRICS}
    if len(by_seed)>1:
        values.update({key+'_std':float(np.std([v[key] for v in by_seed.values()],ddof=0)) for key in METRICS})
    return values,by_seed


def summarize_recipe(recipe):
    fields=['architecture','implementation_type','official_repo','official_commit','model_sha256','adapter_sha256',
            'variant_label','compatibility','sensor_adapter','checkpoint_provenance','checkpoint_sha256',
            'calibration','input_adapter_revision','input_channels','channel_indices','stage1','stage2','input','text',
            'virtual_pretrain_epochs','synthetic_sha256','alignment_checkpoint_sha256']
    return {key:recipe[key] for key in fields if key in recipe}


def main():
    import sys
    sys.path.insert(0,str(HERE))
    from audit_results_20261111 import verify
    OUT.mkdir(exist_ok=True)
    snapshot=read(HERE/'report.json');write(OUT/'continuation_snapshot.json',snapshot)
    readiness=read(HERE/'readiness.json');records=[];audit=[];omitted=[]
    for cell in snapshot['cells']:
        method,dataset=cell['method'],cell['dataset'];base=OLD if cell['status']=='complete_existing_raw_v2' else HERE
        candidates=[]
        for seed in SEEDS:
            expected={f.stem for f in (base/'manifests/splits'/dataset/f'seed_{seed}').glob('*.npz')}
            assert len(expected)==DATASETS[dataset][0]
            files={f.parent.name:f for f in (base/'runs'/method/dataset/f'seed_{seed}').glob('*/result.json') if read(f).get('status')=='complete'}
            if set(files)==expected:candidates.extend((seed,files[fold]) for fold in sorted(expected))
        if not candidates:
            omitted.append(dict(method=method,dataset=dataset,completed_folds=cell['completed_folds'],reason='No complete all-subject seed available at the snapshot; paper fallback remains.'))
            continue
        rows=[]
        for seed,file in candidates:
            checked=verify(file);audit.append(checked);row=read(file)
            rows.append(dict(seed=seed,fold=row['fold'],test=row['test'],result_sha256=checked['result_sha256'],path=str(file)))
        metrics,per_seed=aggregate(rows);included=[seed for seed in SEEDS if str(seed) in per_seed]
        recipe=read(candidates[0][1])['recipe']
        record=dict(id=f'measured-20261119:{method}:{dataset}',method=method,dataset=dataset,
            family='realworld' if dataset=='realworld_har' else dataset,num_classes=DATASETS[dataset][1],
            status='complete' if len(included)==5 else 'provisional_complete_seeds',run_kind='supervised',
            primary_metric='accuracy',metrics=metrics,seed_metrics=per_seed,seeds=included,required_seeds=SEEDS,
            complete_subjects_per_seed=DATASETS[dataset][0],completed_folds=len(rows),required_folds=5*DATASETS[dataset][0],
            observed_completed_folds=cell['completed_folds'],unused_incomplete_folds=max(0,cell['completed_folds']-len(rows)),
            aggregation='Equal test-subject weight within each complete seed; equal seed weight; population SD across complete seeds.',
            protocol='Official UCI test split with a training-only validation split.' if dataset=='uci_har' else 'All-subject LOSO; next numeric subject held out for validation; training-only imputation and normalization.',
            calibration='Canonical measured candidate; independent exact paper reproduction/calibration remains unestablished.',
            dataset_manifest_sha256=read(candidates[0][1])['dataset_manifest_sha256'],recipe=summarize_recipe(recipe),
            variant_label=readiness.get(method,{}).get('variant_label',''),verified=True,fold_results=rows)
        records.append(record)
        print('VERIFIED',method,dataset,len(included),'seeds',len(rows),'folds',flush=True)
    # A deterministic frozen model is evaluated once, never copied five times.
    for dataset,(fold_count,classes) in DATASETS.items():
        root=HERE/'zero_shot_runs/onellm'/dataset/'seed_42'
        files={f.parent.name:f for f in root.glob('*/result.json') if read(f).get('status')=='complete_zero_shot'}
        expected={f.stem for f in (HERE/'manifests/splits'/dataset/'seed_42').glob('*.npz')}
        if set(files)!=expected:
            omitted.append(dict(method='onellm',dataset=dataset,reason='Complete zero-shot subject coverage not yet available.'));continue
        assert len(files)==fold_count
        y=np.load(HERE/'data'/dataset/'y.npy');rows=[]
        for fold in sorted(files):
            file=files[fold];row=read(file);pred=np.load(file.parent/'predictions.npz')
            with np.load(HERE/'manifests/splits'/dataset/'seed_42'/f'{fold}.npz') as split:
                assert np.array_equal(pred['test_indices'],split['test'])
                assert not set(split['train'])&set(split['test'])
            assert np.array_equal(pred['y_true'],y[pred['test_indices']])
            logits=pred['logits'];assert logits.shape==(len(pred['y_true']),classes) and np.isfinite(logits).all()
            guessed=logits.argmax(1);assert np.array_equal(guessed,pred['y_pred'])
            actual=dict(accuracy=float(accuracy_score(pred['y_true'],guessed)),
                f1_macro=float(f1_score(pred['y_true'],guessed,labels=list(range(classes)),average='macro',zero_division=0)),
                f1_weighted=float(f1_score(pred['y_true'],guessed,labels=list(range(classes)),average='weighted',zero_division=0)))
            for key,value in row['test'].items():assert abs(value-actual[key])<1e-10
            entry=dict(seed=42,fold=fold,test=actual,path=str(file),result_sha256=sha(file));rows.append(entry);audit.append(entry)
        metrics,per_seed=aggregate(rows)
        records.append(dict(id=f'measured-20261119:onellm:{dataset}',method='onellm',dataset=dataset,
            family='realworld' if dataset=='realworld_har' else dataset,num_classes=classes,status='complete_zero_shot',run_kind='zero_shot',
            primary_metric='accuracy',metrics=metrics,seed_metrics=per_seed,seeds=[42],required_seeds=[42],
            complete_subjects_per_seed=fold_count,completed_folds=fold_count,required_folds=fold_count,
            aggregation='One deterministic frozen-model pass; equal weight across all held-out subjects. No five-fit SD is asserted.',
            protocol='Author frozen OneLLM checkpoint; zero-shot class-letter readout on the canonical target splits; train-only median imputation.',
            calibration='Task extension from the original generative model; not the supervised OneLLM setting reported by MOSAIC.',
            recipe=summarize_recipe(row['recipe']),variant_label='Frozen author checkpoint · zero-shot HAR adapter',verified=True,fold_results=rows))
        print('VERIFIED onellm',dataset,'zero-shot',fold_count,'folds',flush=True)
    assert len({(r['method'],r['dataset']) for r in records})==len(records)
    value=dict(version='measured-first-20261119-v1',updated_at=datetime.now(timezone.utc).isoformat(),
        snapshot_at=snapshot['updated_at'],selection_policy='Measured results first, independent of their value or paper gap; retain the existing fixed-priority paper result only when no eligible measured aggregate exists.',
        provisional_policy='Include every available complete all-subject seed. Exclude unfinished subject sets; mark fewer than five seeds provisional.',
        overall_metric='accuracy',dataset_weights={'uci_har':.25,'realworld':.25,'pamap2':.25,'usc_had':.25},
        records=records,omitted=omitted,totals=dict(records=len(records),methods=len({r['method'] for r in records}),
            complete_supervised=sum(r['status']=='complete' for r in records),provisional=sum(r['status']=='provisional_complete_seeds' for r in records),zero_shot=sum(r['run_kind']=='zero_shot' for r in records),verified_fold_records=len(audit)),
        paid_api_calls=0,excluded_smoke_and_mock=True,audit=dict(status='passed',predictions_and_metrics_recomputed=True,
            weights_splits_sources_sha_verified_for_supervised=True,original_dataset_arrays_sha_verified=True))
    write(OUT/'measured_results_20261119.json',value)
    write(OUT/'artifact_audit_20261119.json',dict(status='passed',verified=audit,snapshot_at=snapshot['updated_at']))
    print('EXPORT COMPLETE',json.dumps(value['totals']),flush=True)


if __name__=='__main__':main()
