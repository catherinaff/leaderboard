"""Raw, recording-preserving extensions for the November HAR continuation."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.io import loadmat

from prepare import HERE, ROOT, RAW, SEEDS, atomic_json, sha, splits

POSITIONS=['chest','forearm','head','shin','thigh','upperarm','waist']
ACTIVITIES=['climbingdown','climbingup','jumping','lying','running','sitting','standing','walking']


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


def csv_members(archive, trail='', session=1):
    """Preserve the repeated sessions in nested official RealWorld archives."""
    for member in archive.namelist():
        name=Path(member).name
        if name.startswith('.') or '__MACOSX' in member:
            continue
        full=trail+member
        if name.lower().endswith('.zip'):
            match=re.search(r'_(\d+)_csv\.zip$',name,re.I)
            nested_session=int(match[1]) if match else session
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as child:
                yield from csv_members(child,full+'!',nested_session)
        elif name.lower().endswith('.csv'):
            yield full,name,session,archive.read(member)


def realworld_recordings():
    base=RAW/'realworld_har_csv_zips'
    for sid in range(1,16):
        for activity in ACTIVITIES:
            streams={};sources=[]
            for sensor in ['acc','gyr']:
                path=base/f's{sid:02d}'/f'{sensor}_{activity}_csv.zip'
                with zipfile.ZipFile(path) as archive:
                    assert archive.testzip() is None,('CRC failure',path)
                    sources.append(dict(archive=str(path),sha256=sha(path)))
                    for member,name,session,raw in csv_members(archive):
                        pattern=r'^(?:acc|gyr|Gyroscope)_'+activity+r'(?:_(\d+))?_('+('|'.join(POSITIONS))+r')\.csv$'
                        match=re.fullmatch(pattern,name,re.I)
                        if not match:raise ValueError(('Unknown official sensor CSV',path,member))
                        explicit,position=match.groups()
                        if explicit:
                            assert int(explicit)==session or session==1
                            session=int(explicit)
                        key=(session,position.lower(),sensor)
                        assert key not in streams,('Duplicate stream; refusing to choose a session by size',key,path,member)
                        frame=pd.read_csv(io.BytesIO(raw),usecols=['attr_time','attr_x','attr_y','attr_z'])
                        frame=frame.apply(pd.to_numeric,errors='coerce').dropna(subset=['attr_time'])
                        frame=frame.sort_values('attr_time',kind='stable').drop_duplicates('attr_time',keep='first')
                        assert len(frame)>1,(path,member)
                        streams[key]=(frame,dict(member=member,samples=len(frame),csv_sha256=hashlib.sha256(raw).hexdigest()))
            for session in sorted({key[0] for key in streams}):
                available={(pos,sensor):streams[(ss,pos,sensor)] for ss,pos,sensor in streams if ss==session}
                assert any(sensor=='acc' for pos,sensor in available) and any(sensor=='gyr' for pos,sensor in available)
                # All devices share the original recording clock. Do not zero
                # each stream independently: it would shift acc/gyro alignment.
                begin=max(float(f.attr_time.iloc[0]) for f,_ in available.values())
                end=min(float(f.attr_time.iloc[-1]) for f,_ in available.values())
                assert end>begin,(sid,activity,session,begin,end)
                grid=np.arange(0,(end-begin)/1000,.02,dtype=np.float64)
                valid=np.ones(len(grid),bool);channels=[];missing=[];members={}
                for position in POSITIONS:
                    for sensor in ['acc','gyr']:
                        if (position,sensor) not in available:
                            channels.extend([np.full(len(grid),np.nan,np.float32)]*3)
                            missing.append(f'{position}:{sensor}')
                            continue
                        frame,meta=available[(position,sensor)]
                        t=(frame.attr_time.to_numpy(np.float64)-begin)/1000
                        assert np.all(np.diff(t)>0)
                        ii=np.searchsorted(t,grid).clip(0,len(t)-1);prev=(ii-1).clip(0)
                        ii=np.where(np.abs(t[ii]-grid)<np.abs(t[prev]-grid),ii,prev)
                        valid&=np.abs(t[ii]-grid)<=.1
                        for axis in ['attr_x','attr_y','attr_z']:
                            values=frame[axis].to_numpy(np.float64)
                            channels.append(np.interp(grid,t,values).astype(np.float32))
                        members[f'{position}:{sensor}']=meta
                x=np.stack(channels,axis=1)
                edges=np.r_[0,np.flatnonzero(np.diff(valid.astype(np.int8)))+1,len(grid)]
                kept=0
                for block,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                    if hi-lo<100 or not valid[lo]:continue
                    kept+=1
                    rid=f's{sid:02d}/{activity}/session_{session}/block_{block}'
                    source=dict(archives=sources,members=members,session=session,
                        original_start_ms=begin+grid[lo]*1000,original_end_ms=begin+grid[hi-1]*1000,
                        missing_streams=missing,missing_policy='Missing devices remain NaN for training-fold-only median imputation; no zero-filled sensors.',
                        alignment='Original absolute millisecond timestamps; common intersection at 50 Hz; linear interpolation only inside contiguous intervals; nearest source sample within 100 ms for every available stream.',
                        channel_order=[f'{pos}:{sensor}:{axis}' for pos in POSITIONS for sensor in ['acc','gyr'] for axis in ['x','y','z']])
                    yield sid,rid,x[lo:hi],np.full(hi-lo,ACTIVITIES.index(activity),np.int64),source
                assert kept>0,('No full two-second aligned window in source session',sid,activity,session)
                print('aligned',sid,activity,session,'blocks',kept,'missing',missing,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['usc_had','realworld_har'],required=True);args=p.parse_args()
    if args.dataset=='usc_had':
        write_dataset('usc_had',usc_recordings(),dict(archive=str(RAW/'usc_had.zip'),archive_sha256=sha(RAW/'usc_had.zip'),native_rate_hz=100,classes=12,subjects=14,fold_policy='LOSO, next sorted subject for validation, remaining 12 train subjects, five fixed seeds.'))
    else:
        write_dataset('realworld_har',realworld_recordings(),dict(source='Official Mannheim RealWorld acc/gyr CSV ZIP distribution',native_rate_hz=50,classes=8,subjects=15,channel_order=POSITIONS,session_policy='Keep every separate session, including nested climbing archives; never choose only the longest recording.',fold_policy='LOSO, next sorted subject for validation, remaining 13 subjects train, five fixed seeds.'))


if __name__=='__main__':main()
