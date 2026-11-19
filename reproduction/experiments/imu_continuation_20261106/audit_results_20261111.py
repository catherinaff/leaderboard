"""Verify saved weights, frozen splits, predictions and recomputed metrics."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'imu_reproduction_20261028'


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.audit-tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


@lru_cache(maxsize=10)
def data(base,dataset):
    directory=Path(base)/'data'/dataset
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['arrays_sha256'].items():
        assert sha(directory/f'{name}.npy')==digest,('Dataset array changed',dataset,name)
    return (sha(directory/'manifest.json'),np.load(directory/'y.npy'),
        np.load(directory/'subjects.npy'),len(manifest['classes']))


def verify(file):
    row=json.loads(file.read_text());out=file.parent
    assert row['status']=='complete'
    assert row['seed'] in [42,123,2021,2024,3407]
    base=OLD if file.is_relative_to(OLD) else HERE
    fingerprint,y,subjects,classes=data(str(base),row['dataset'])
    assert fingerprint==row['dataset_manifest_sha256'],'Dataset manifest mismatch'
    for name,digest in row['artifacts'].items():assert sha(out/name)==digest,('Artifact mismatch',name)
    for name,digest in row.get('execution_source_sha256',{}).items():assert sha(out/'execution_source'/name)==digest,('Execution source mismatch',name)
    if row['recipe'].get('pretrain_sha256'):
        assert sha(out/'pretrain_best.pt')==row['recipe']['pretrain_sha256']
    central=base/'manifests/splits'/row['dataset']/f"seed_{row['seed']}"/f"{row['fold']}.npz"
    assert sha(central)==row['split_sha256'],'Frozen split mismatch'
    with np.load(central) as canonical,np.load(out/'split.npz') as saved,np.load(out/'predictions.npz') as pred:
        tr,va,te=[saved[name] for name in ['train','validation','test']]
        for name in ['train','validation','test']:assert np.array_equal(saved[name],canonical[name]),('Wrong split indices',name)
        assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
        if row['dataset']!='uci_har':
            assert not set(subjects[tr])&set(subjects[va]) and not set(subjects[tr])&set(subjects[te]) and not set(subjects[va])&set(subjects[te])
        assert np.array_equal(te,pred['test_indices']) and np.array_equal(y[te],pred['y_true'])
        assert row['counts']==dict(train=len(tr),validation=len(va),test=len(te))
        scores=pred['logits'] if 'logits' in pred else pred['probabilities']
        assert scores.shape==(len(te),classes) and np.isfinite(scores).all()
        if 'probabilities' in pred:assert np.allclose(scores.sum(1),1,atol=1e-5) and (scores>=0).all()
        guesses=scores.argmax(1);assert np.array_equal(guesses,pred['y_pred'])
        actual=dict(accuracy=float(accuracy_score(y[te],guesses)),
            f1_macro=float(f1_score(y[te],guesses,labels=list(range(classes)),average='macro',zero_division=0)),
            f1_weighted=float(f1_score(y[te],guesses,labels=list(range(classes)),average='weighted',zero_division=0)))
        for name,value in actual.items():assert abs(value-row['test'][name])<1e-10,('Wrong test metric',name)
        assert np.array_equal(confusion_matrix(y[te],guesses,labels=list(range(classes))),row['confusion_matrix'])
        validation=pred['validation_logits'] if 'validation_logits' in pred else pred['validation_probabilities']
        assert np.array_equal(va,pred['validation_indices']) and np.array_equal(y[va],pred['validation_y_true'])
        assert validation.shape==(len(va),classes) and np.isfinite(validation).all()
        for name,average in [('f1_macro','macro'),('f1_weighted','weighted')]:
            value=float(f1_score(y[va],validation.argmax(1),labels=list(range(classes)),average=average,zero_division=0))
            assert abs(value-row['validation'][name])<1e-10,('Wrong validation metric',name)
        assert abs(float(accuracy_score(y[va],validation.argmax(1)))-row['validation']['accuracy'])<1e-10
    if row['method'] in ['hart','mobilehart'] and row['dataset']=='uci_har':
        assert row['recipe'].get('input_adapter_revision')=='uci-total-acc-20261111'
        with np.load(out/'scaler.npz') as scaler:assert scaler['channel_indices'].tolist()==[6,7,8,3,4,5]
    return dict(path=str(file),result_sha256=sha(file),method=row['method'],dataset=row['dataset'],seed=row['seed'],fold=row['fold'])


def main():
    os.nice(10);started=now()
    files=sorted(HERE.glob('runs/*/*/seed_*/*/result.json'))
    for method in ['deepconvlstm','if_convtransformer','unimts','limu_bert']:
        for dataset in ['uci_har','pamap2']:files.extend(sorted((OLD/'runs'/method/dataset).glob('seed_*/*/result.json')))
    # Check dataset fingerprints once before parallel per-run file validation.
    for file in files:
        row=json.loads(file.read_text());base=OLD if file.is_relative_to(OLD) else HERE
        data(str(base),row['dataset'])
    passed=[];errors=[]
    summary=dict(status='running',started_at=started,total=len(files),verified=0,failed=0)
    write(HERE/'state/artifact_audit.json',summary)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(verify,file):file for file in files}
        for future in as_completed(tasks):
            try:passed.append(future.result())
            except Exception as e:errors.append(dict(path=str(tasks[future]),error=f'{type(e).__name__}: {e}'))
            if (len(passed)+len(errors))%50==0:
                summary.update(verified=len(passed),failed=len(errors),updated_at=now())
                write(HERE/'state/artifact_audit.json',summary)
                print('audit',len(passed)+len(errors),'/',len(files),'failed',len(errors),flush=True)
    report=dict(started_at=started,completed_at=now(),scope='New complete runs plus the eight reused raw-v2 cells; superseded body_acc variants are archived separately.',
        verified=passed,errors=errors)
    write(HERE/'audits/artifact_audit_20261111.json',report)
    summary.update(status='passed' if not errors else 'requires_repair',verified=len(passed),failed=len(errors),updated_at=now(),
        report=str(HERE/'audits/artifact_audit_20261111.json'),errors=errors)
    write(HERE/'state/artifact_audit.json',summary)
    print('AUDIT COMPLETE',summary,flush=True)


if __name__=='__main__':main()
