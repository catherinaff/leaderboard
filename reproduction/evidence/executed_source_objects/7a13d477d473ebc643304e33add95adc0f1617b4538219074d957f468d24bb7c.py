"""Raw, recording-preserving extensions for the November HAR continuation."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import numpy as np
import pandas as pd
from scipy.io import loadmat

from prepare import HERE, ROOT, RAW, SEEDS, atomic_json, sha, splits


def write_dataset(name, records, provenance):
    dest=HERE/'data'/name
    if (dest/'manifest.json').exists():
        manifest=json.loads((dest/'manifest.json').read_text())
        for key,h in manifest['arrays_sha256'].items():
            assert sha(dest/f'{key}.npy')==h
        print('Already prepared and verified',name,flush=True)
        return manifest
    if shutil.disk_usage(HERE).free<68*2**30:
        raise RuntimeError('Need 68 GiB free before preparing a dataset; reserve is 64 GiB')
    bench=json.loads((ROOT/'benchmarks'/f'{name}.json').read_text())
    window=bench['protocol']['window_size_samples'];stride=bench['protocol']['stride_samples']
    xs=[];ys=[];subjects=[];recording_ids=[];starts=[];meta=[]
    for sid,rid,values,labels,source in records:
        assert values.shape[1]==bench['channels'] and len(values)==len(labels)
        # Keep leading missing samples for the fold's train-only imputer.
        values=pd.DataFrame(np.where(np.isfinite(values),values,np.nan)).ffill().to_numpy(np.float32)
        n=0
        for start in range(0,len(values)-window+1,stride):
            local=labels[start:start+window]
            assert np.all(local==local[0]),'Activity/recording boundary must not be crossed'
            xs.append(values[start:start+window]);ys.append(int(local[0]));subjects.append(sid)
            recording_ids.append(len(meta));starts.append(start);n+=1
        meta.append(dict(recording=rid,subject=sid,samples=len(values),windows=n,source=source))
    arrays={'x':np.stack(xs).astype(np.float32),'y':np.asarray(ys,np.int64),
        'subjects':np.asarray(subjects,np.int64),'recordings':np.asarray(recording_ids,np.int64),
        'starts':np.asarray(starts,np.int64)}
    assert arrays['x'].shape[1:]==(window,bench['channels'])
    assert set(arrays['y'])==set(range(len(bench['classes'])))
    assert len(set(subjects))==bench['subjects']
    dest.mkdir(parents=True,exist_ok=True)
    for key,arr in arrays.items():np.save(dest/f'{key}.npy',arr,allow_pickle=False)
    manifest=dict(dataset=name,version='continuation-raw-20261106',shape=list(arrays['x'].shape),
        subjects=sorted(set(subjects)),classes=bench['classes'],recordings=meta,
        preprocessing_sha256=sha(__file__),benchmark_sha256=sha(ROOT/'benchmarks'/f'{name}.json'),
        missing='Recording-local forward fill, leading NaNs retained; fold-only median and z-score in train.py.',
        windowing='Native benchmark windows; no recording/activity/session boundary crossing.',
        arrays_sha256={key:sha(dest/f'{key}.npy') for key in arrays},provenance=provenance)
    atomic_json(dest/'manifest.json',manifest)
    for seed in SEEDS:
        for fold,tr,va,te in splits(name,seed):
            assert len(tr)>0 and len(va)>0 and len(te)>0
            s=arrays['subjects'];assert not set(s[tr])&set(s[va]) and not set(s[tr])&set(s[te]) and not set(s[va])&set(s[te])
            out=HERE/'manifests/splits'/name/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'{fold}.npz',train=tr,validation=va,test=te)
    print('PREPARED',name,manifest['shape'],'subjects',manifest['subjects'],flush=True)
    return manifest


def usc_recordings():
    base=RAW/'usc_had/USC-HAD'
    paths=sorted(base.glob('Subject*/a*t*.mat'))
    assert len(paths)==840,('Expected 14 subjects x 12 activities x 5 trials',len(paths))
    seen=set()
    for path in paths:
        sid=int(re.fullmatch(r'Subject(\d+)',path.parent.name)[1])
        activity,trial=map(int,re.fullmatch(r'a(\d+)t(\d+)',path.stem).groups())
        assert 1<=sid<=14 and 1<=activity<=12 and 1<=trial<=5
        assert (sid,activity,trial) not in seen;seen.add((sid,activity,trial))
        data=loadmat(path)
        assert int(str(data['subject'].reshape(-1)[0]))==sid
        # The original Subject13/a11t4.mat misspells this metadata key.
        activity_key='activity_number' if 'activity_number' in data else 'activity_numbr'
        metadata_activity=int(str(data[activity_key].reshape(-1)[0]))
        x=np.asarray(data['sensor_readings'],dtype=np.float32)
        assert x.ndim==2 and x.shape[1]==6 and x.shape[0]>=200
        yield sid,str(path.relative_to(base)),x,np.full(len(x),activity-1,np.int64),dict(path=str(path),sha256=sha(path),sensor_order='acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z',activity_metadata_key=activity_key,filename_activity=activity,metadata_activity=metadata_activity,label_metadata_discrepancy=metadata_activity!=activity,label_policy='Use official activity filename, matching the existing USC benchmark loader; retain metadata conflicts in provenance rather than silently relabeling or excluding trials.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['usc_had'],required=True);args=p.parse_args()
    if args.dataset=='usc_had':
        write_dataset('usc_had',usc_recordings(),dict(archive=str(RAW/'usc_had.zip'),archive_sha256=sha(RAW/'usc_had.zip'),native_rate_hz=100,classes=12,subjects=14,fold_policy='LOSO, next sorted subject for validation, remaining 12 train subjects, five fixed seeds.'))


if __name__=='__main__':main()
