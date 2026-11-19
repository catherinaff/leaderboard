"""Raw-data-only Standard preparation. Does not estimate any population statistic.

Every retained window carries recording/start provenance. Leading missing values
remain NaN; the training fold alone supplies the imputation median and scaler.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'experiments/datasets'
SEEDS = [42, 123, 2021, 2024, 3407]
OPP_CODES = [406516,406517,404516,404517,406520,404520,406505,404505,
             406519,406511,406508,404519,404511,404508,408512,407521,405506]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def raw_recordings(name):
    """Yield subject, immutable recording id, raw values and per-sample labels."""
    if name == 'pamap2':
        codes = [1,2,3,4,5,6,7,12,13,16,17,24]
        cols = [i for start in [3,20,37] for i in
                [*range(start+1,start+4),*range(start+7,start+10),*range(start+10,start+13)]]
        for p in sorted((RAW/'pamap2/PAMAP2_Dataset/Protocol').glob('subject*.dat')):
            sid = int(p.stem[-3:])
            if sid not in range(101,109):
                continue
            a = np.loadtxt(p, dtype=np.float32)
            y = np.full(len(a), -1, np.int64)
            for i,c in enumerate(codes):
                y[a[:,1] == c] = i
            yield sid, p.name, a[:,cols], y
    elif name == 'opportunity_a':
        deletes = np.concatenate([np.arange(a,b) for a,b in
                    [(46,50),(59,63),(72,76),(85,89),(98,102),(134,243),(244,249)]])
        for p in sorted((RAW/'opportunity/OpportunityUCIDataset/dataset').glob('S*-ADL*.dat')):
            a = np.delete(np.loadtxt(p, dtype=np.float32), deletes, axis=1)
            assert a.shape[1] == 116, a.shape
            y = np.full(len(a), -1, np.int64)
            for i,c in enumerate(OPP_CODES):
                y[a[:,115] == c] = i
            yield int(p.name[1]), p.name, a[:,1:114], y
    elif name == 'wisdm':
        labels = {s:i for i,s in enumerate(['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing'])}
        rows = []
        sid0 = None
        t0 = None
        rid = 0
        p = RAW/'wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
        def unpack(rows):
            a = np.asarray(rows, np.float64)
            # Author records nominal 20 Hz. Raw timestamps contain zeros and
            # repeated/non-monotonic values, so retain native sample order;
            # timestamp interpolation would silently discard most subjects.
            return a[:,2:].astype(np.float32), a[:,1].astype(np.int64)
        for line in p.open(errors='replace'):
            parts = line.strip().rstrip(';').split(',')
            if len(parts) != 6:
                continue
            try:
                sid,t = int(parts[0]),int(parts[2])
                v = [float(x) for x in parts[3:]]
                label = labels[parts[1]]
            except (ValueError,KeyError):
                continue
            if rows and (sid != sid0 or (t>0 and t0 and abs(t-t0)>2_000_000_000)):
                if len(rows)>1:
                    x,y=unpack(rows)
                    yield sid0, f'wisdm_block_{rid}', x,y
                rid += 1
                rows=[]
            sid0 = sid
            if t>0:
                t0=t
            rows.append([t,label,*v])
        if len(rows)>1:
            x,y=unpack(rows)
            yield sid0,f'wisdm_block_{rid}',x,y
    elif name == 'hhar':
        base=RAW/'hhar/Activity recognition exp'
        cols=['Creation_Time','x','y','z','User','Device','gt']
        acc=pd.read_csv(base/'Phones_accelerometer.csv',usecols=cols)
        gyr=pd.read_csv(base/'Phones_gyroscope.csv',usecols=cols)
        users={u:i+1 for i,u in enumerate(sorted(acc.User.unique()))}
        labels={s:i for i,s in enumerate(['bike','sit','stand','walk','stairsup','stairsdown'])}
        gg={k:g for k,g in gyr.groupby(['User','Device'],sort=True)}
        for (user,device),g in acc.groupby(['User','Device'],sort=True):
            if (user,device) not in gg:
                continue
            # Keep time gaps and clock resets separate before sorting/alignment.
            times=g.Creation_Time.to_numpy(np.int64)
            cuts=np.r_[0,np.flatnonzero(np.abs(np.diff(times))>1_000_000_000)+1,len(g)]
            gy=gg[(user,device)].sort_values('Creation_Time').drop_duplicates('Creation_Time')
            for block,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
                a=g.iloc[lo:hi].sort_values('Creation_Time').drop_duplicates('Creation_Time')
                if len(a)<2:
                    continue
                start,end=int(a.Creation_Time.iloc[0]),int(a.Creation_Time.iloc[-1])
                b=gy[(gy.Creation_Time>=start-50_000_000)&(gy.Creation_Time<=end+50_000_000)]
                if len(b)<2:
                    continue
                begin=max(start,int(b.Creation_Time.iloc[0]))
                end=min(end,int(b.Creation_Time.iloc[-1]))
                if end<=begin:
                    continue
                grid=np.arange(0,(end-begin)*1e-9,.02)
                ta=(a.Creation_Time.to_numpy(np.int64)-begin)*1e-9
                tb=(b.Creation_Time.to_numpy(np.int64)-begin)*1e-9
                ia=np.searchsorted(ta,grid).clip(0,len(ta)-1)
                prev=(ia-1).clip(0)
                ia=np.where(np.abs(ta[ia]-grid)<np.abs(ta[prev]-grid),ia,prev)
                ib=np.searchsorted(tb,grid).clip(0,len(tb)-1)
                prev=(ib-1).clip(0)
                ib=np.where(np.abs(tb[ib]-grid)<np.abs(tb[prev]-grid),ib,prev)
                valid=(np.abs(ta[ia]-grid)<=.05)&(np.abs(tb[ib]-grid)<=.05)
                x=np.stack([np.interp(grid,t,df[c]) for df,t in [(a,ta),(b,tb)] for c in ['x','y','z']],1).astype(np.float32)
                y=a['gt'].map(labels).fillna(-1).to_numpy(np.int64)[ia]
                # Missing entire intervals must not be concatenated into windows.
                edges=np.r_[0,np.flatnonzero(np.diff(valid.astype(int)))+1,len(valid)]
                for k,(i,j) in enumerate(zip(edges[:-1],edges[1:])):
                    if valid[i]:
                        yield users[user],f'{user}_{device}_{block}_{k}',x[i:j],y[i:j]
    else:
        raise ValueError(name)


def prepare(name):
    dest=HERE/'data'/name
    if (dest/'manifest.json').exists():
        print('prepared',name,flush=True)
        return
    dest.mkdir(parents=True,exist_ok=True)
    bench=json.loads((ROOT/'benchmarks'/f'{name}.json').read_text())
    xs,ys,ss,rr,starts=[],[],[],[],[]
    recordings=[]
    if name=='uci_har':
        signals=['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z']
        for split in ['train','test']:
            b=RAW/'uci_har/UCI HAR Dataset'/split
            x=np.stack([np.loadtxt(b/'Inertial Signals'/f'{s}_{split}.txt',dtype=np.float32) for s in signals],2)
            y=np.loadtxt(b/f'y_{split}.txt',dtype=np.int64)-1
            s=np.loadtxt(b/f'subject_{split}.txt',dtype=np.int64)
            xs.extend(x);ys.extend(y);ss.extend(s);rr.extend([len(recordings)]*len(y));starts.extend(range(len(y)))
            recordings.append({'name':split,'samples':len(y),'windows':len(y)})
    else:
        w,stride=bench['protocol']['window_size_samples'],bench['protocol']['stride_samples']
        for subject,rid,x,y in raw_recordings(name):
            x=pd.DataFrame(np.where(np.isfinite(x),x,np.nan)).ffill().to_numpy(np.float32)
            count=0
            for start in range(0,len(x)-w+1,stride):
                labels=y[start:start+w];labels=labels[labels>=0]
                if not len(labels):
                    continue
                values,counts=np.unique(labels,return_counts=True)
                k=counts.argmax()
                if counts[k]*2 < len(labels):
                    continue
                xs.append(x[start:start+w]);ys.append(values[k]);ss.append(subject);rr.append(len(recordings));starts.append(start)
                count+=1
            recordings.append({'name':rid,'subject':int(subject),'samples':len(x),'windows':count})
    arrays={'x':np.stack(xs).astype(np.float32),'y':np.asarray(ys,np.int64),
            'subjects':np.asarray(ss,np.int64),'recordings':np.asarray(rr,np.int64),'starts':np.asarray(starts,np.int64)}
    assert arrays['x'].shape[2]==bench['channels']
    assert set(arrays['y'])==set(range(len(bench['classes'])))
    for key,a in arrays.items():
        np.save(dest/f'{key}.npy',a,allow_pickle=False)
    manifest={'dataset':name,'version':'standard-raw-v2-20261028','shape':list(arrays['x'].shape),
              'subjects':sorted(set(map(int,ss))),'classes':bench['classes'],'recordings':recordings,
              'preprocessing_sha256':sha(__file__),'benchmark_sha256':sha(ROOT/'benchmarks'/f'{name}.json'),
              'missing':'recording-local forward fill; leading NaNs retained for train-only median',
              'window_label':'majority non-null, at least 50% support among non-null; no recording crossing',
              'arrays_sha256':{k:sha(dest/f'{k}.npy') for k in arrays}}
    archive={'uci_har':'uci_har.zip','wisdm':'wisdm_ar_latest.tar.gz','pamap2':'pamap2.zip','hhar':'hhar.zip','opportunity_a':'opportunity.zip'}[name]
    manifest['archive_sha256']=sha(RAW/archive)
    atomic_json(dest/'manifest.json',manifest)
    print(name,manifest['shape'],manifest['subjects'],flush=True)


def splits(name,seed):
    from sklearn.model_selection import train_test_split
    base=HERE/'data'/name
    y=np.load(base/'y.npy');subjects=np.load(base/'subjects.npy');idx=np.arange(len(y))
    if name=='uci_har':
        rec=np.load(base/'recordings.npy')
        tr,va=train_test_split(idx[rec==0],test_size=.1,stratify=y[rec==0],random_state=seed)
        yield 'official',tr,va,idx[rec==1]
    elif name=='wisdm':
        yield 'fixed_subjects',idx[subjects<=28],idx[np.isin(subjects,[29,30])],idx[subjects>=31]
    else:
        unique=sorted(set(subjects))
        for i,s in enumerate(unique):
            val=unique[(i+1)%len(unique)]
            yield f'test_{s}_val_{val}',idx[~np.isin(subjects,[s,val])],idx[subjects==val],idx[subjects==s]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['uci_har','wisdm','pamap2','hhar','opportunity_a'])
    args=p.parse_args()
    for name in args.datasets:
        prepare(name)
        for seed in SEEDS:
            for fold,tr,va,te in splits(name,seed):
                assert len(tr) and len(va) and len(te)
                assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
                d=HERE/'manifests/splits'/name/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(d/f'{fold}.npz',train=tr,validation=va,test=te)
