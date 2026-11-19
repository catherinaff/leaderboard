"""Verify native smoke/zero-shot evidence; never rank a plumbing fixture."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from api_adapters import atomic_json

p=Path(__file__).resolve().parent
rows=[]
for directory in ['smoke_zero_shot/onellm','smoke_large/sensorllm','zero_shot_runs/onellm']:
 for file in sorted((p/directory).glob('*/*/*/result.json')):
  value=json.loads(file.read_text());dataset=value['dataset']
  meta=json.loads((p/'data'/dataset/'manifest.json').read_text());classes=len(meta['classes'])
  y=np.load(p/'data'/dataset/'y.npy')
  prediction=np.load(file.parent/'predictions.npz')
  assert np.array_equal(y[prediction['test_indices']],prediction['y_true'])
  assert prediction['logits'].shape==(len(prediction['y_true']),classes) and np.isfinite(prediction['logits']).all()
  guess=prediction['logits'].argmax(1);assert np.array_equal(guess,prediction['y_pred'])
  assert abs(accuracy_score(prediction['y_true'],guess)-value['test']['accuracy'])<1e-10
  assert abs(f1_score(prediction['y_true'],guess,labels=list(range(classes)),average='macro',zero_division=0)-value['test']['f1_macro'])<1e-10
  for name,digest in value.get('artifacts',{}).items():assert hashlib.sha256((file.parent/name).read_bytes()).hexdigest()==digest
  rows.append(dict(path=str(file),method=value['method'],dataset=dataset,status=value['status'],sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
corpora=[]
for file in sorted((p/'assets_20261113/imugpt/synthetic').glob('*.npz')):
 value=json.loads(file.with_suffix('.json').read_text());arrays=np.load(file)
 assert value['generation_pipeline']=='author_T2M_GPT_BVH_IMUSim' and value['paid_api_calls']==0
 assert np.isfinite(arrays['x']).all() and set(arrays['y'])==set(range(len(value['classes'])))
 assert hashlib.sha256(file.read_bytes()).hexdigest()==value['npz_sha256']
 corpora.append(dict(dataset=file.stem,shape=list(arrays['x'].shape),classes=len(value['classes']),sha256=value['npz_sha256']))
atomic_json(p/'audits/native_adapter_evidence_20261113.json',dict(status='passed',completed_at=datetime.now(timezone.utc).isoformat(),
    note='Native smoke scores are excluded from ranking. Frozen zero-shot records are separate from independent supervised fits.',verified=rows,verified_genuine_virtual_corpora=corpora,paid_api_calls=0))
print('NATIVE AUDIT PASSED',len(rows),'prediction records;',len(corpora),'genuine virtual corpora',flush=True)
