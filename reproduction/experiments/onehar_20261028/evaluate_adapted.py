"""Exploratory frozen-checkpoint evaluation on available IoTBench datasets.

This is explicitly a custom input adapter, not author-protocol reproduction
or a newly trained LOSO baseline. No labels are used to fit model weights.
"""
from pathlib import Path
import csv
import json
import time
import re
import zipfile
import io

import numpy as np
import torch
from scipy.signal import resample_poly
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def adapt(x, sr):
    """Keep physical time: center crop / edge-pad to 2.4 s, then resample to 50 Hz."""
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] >= 1
    target = round(sr * 2.4)
    if x.shape[2] > target:
        begin = (x.shape[2] - target) // 2
        x = x[:, :, begin:begin + target]
    elif x.shape[2] < target:
        missing = target - x.shape[2]
        x = np.pad(x, ((0,0),(0,0),(missing//2, missing-missing//2),(0,0)), mode='edge')
    if not np.isfinite(x).all():
        for row, sensor, channel in np.argwhere(~np.isfinite(x).all(axis=2)):
            y = x[row, sensor, :, channel]
            good = np.isfinite(y)
            y[:] = np.interp(np.arange(len(y)), np.flatnonzero(good), y[good]) if good.any() else 0
    if sr != 50:
        x = resample_poly(x, 50, sr, axis=2)
    assert x.shape[2:] == (120, 3)
    # The released loader uses one scalar mean/std per sensor position, pooling
    # all test examples/time/channel axes. Retain this transductive convention.
    mean = x.mean(axis=(0,2,3), keepdims=True, dtype=np.float64)
    std = x.std(axis=(0,2,3), keepdims=True, dtype=np.float64)
    if (std <= 0).any():
        raise ValueError('Constant sensor position')
    return ((x-mean)/std).astype(np.float32)


def processed(root, name, channels, sr, positions, candidates, subset=None):
    path = root / 'datasets/processed' / (name + '_windows.npz')
    with np.load(path) as data:
        select = np.ones(len(data['y']), dtype=bool) if subset is None else subset(data['subjects'])
        raw = data['x'][select]
        x = np.stack([raw[:,:,ids] for ids in channels], axis=1)
        return {'dataset':name,'x':adapt(x,sr),'y':data['y'][select],'subjects':data['subjects'][select],
                'classes':data['label_names'].tolist(),'candidates':candidates,'positions':positions,
                'source':str(path),'source_shape':list(raw.shape),'sample_rate_hz':sr,
                'selection':'subjects 31–36' if name=='wisdm' else 'all available subjects; frozen checkpoint; no LOSO retraining'}


def available_cases(root, options):
    uci=root/'datasets/uci_har/UCI HAR Dataset'
    names=[options.uci_acc+'_acc_'+a for a in ['x','y','z']]
    x=np.stack([np.loadtxt(uci/'test/Inertial Signals'/f'{n}_test.txt',dtype=np.float32) for n in names],axis=-1)
    y=np.loadtxt(uci/'test/y_test.txt',dtype=int)-1
    subjects=np.loadtxt(uci/'test/subject_test.txt',dtype=int)
    yield {'dataset':'uci_har','x':adapt(x[:,None],50),'y':y,'subjects':subjects,'classes':['Walking','Walking Upstairs','Walking Downstairs','Sitting','Standing','Laying'],
           'candidates':['walking','upstairs','downstairs','sitting','standing','lying'],'positions':['waist'],
           'source':str(uci/'test'),'source_shape':list(x.shape),'sample_rate_hz':50,'selection':'official UCI-HAR 2947-window test partition','acceleration_input':options.uci_acc+'_acc'}
    yield processed(root,'wisdm',[[0,1,2]],20,['pocket'],['walking','running','upstairs','downstairs','sitting','standing'],lambda s:s>=31)
    yield processed(root,'pamap2',[[9,10,11],[0,1,2],[18,19,20]],100,['chest','wrist','ankle'],['lying','sitting','standing','walking','running','cycling','nordic walking','upstairs','downstairs','vacuum cleaning','ironing','rope jumping'])
    yield processed(root,'hhar',[[0,1,2]],50,['waist'],['cycling','sitting','standing','walking','upstairs','downstairs'])
    yield processed(root,'usc_had',[[0,1,2]],100,['hip'],['walking forward','walking left','walking right','upstairs','downstairs','running','jumping','sitting','standing','lying','elevator up','elevator down'])

    # The common 113-column OPPORTUNITY selection retains five body IMUs.
    # Derive the accelerometer columns from the raw official column numbering.
    delete=np.concatenate([np.arange(a,b) for a,b in [(46,50),(59,63),(72,76),(85,89),(98,102),(134,243),(244,249)]])
    raw_indices=np.delete(np.arange(250),delete)[1:114]
    sensor_indices=[[int(np.flatnonzero(raw_indices==j)[0]) for j in range(start,start+3)] for start in [37,50,63,76,89]]
    positions=['back','right_upper_arm','right_forearm','left_upper_arm','left_forearm']
    yield processed(root,'opportunity_a',sensor_indices,30,positions,['opening door one','opening door two','closing door one','closing door two','opening fridge','closing fridge','opening dishwasher','closing dishwasher','opening drawer one','closing drawer one','opening drawer two','closing drawer two','opening drawer three','closing drawer three','cleaning table','drinking from cup','toggling switch'])
    yield processed(root,'opportunity_b',sensor_indices,30,positions,['standing','walking','sitting','lying'])
    limu=root/'repos/LIMU-BERT-Public/dataset'
    for folder,name,candidates in [('motion','motionsense',['downstairs','upstairs','sitting','standing','walking','running']),('shoaib','shoaib',['walking','sitting','standing','running','cycling','upstairs','downstairs'])]:
        x=np.load(limu/folder/'data_20_120.npy')[:,:,:3]
        labels=np.load(limu/folder/'label_20_120.npy')
        yield {'dataset':name,'x':adapt(x[:,None],20),'y':labels[:,0,0].astype(int),'subjects':labels[:,0,1 if folder=='motion' else 2].astype(int),
               'classes':candidates,'candidates':candidates,'positions':['pocket' if folder=='motion' else 'five source placements evaluated as separate windows'],
               'source':str(limu/folder),'source_shape':list(x.shape),'sample_rate_hz':20,
               'selection':'all released LIMU-BERT processed windows; exploratory, no canonical split claimed'}

    artifacts=Path(__file__).resolve().parent/'artifacts'
    if (artifacts/'mhealth.zip').exists():
        windows=[]; targets=[]; people=[]
        with zipfile.ZipFile(artifacts/'mhealth.zip') as archive:
            for member in sorted(archive.namelist()):
                match=re.search(r'mHealth_subject(\d+)\.log$',member)
                if not match: continue
                with archive.open(member) as stream: raw=np.loadtxt(stream,dtype=np.float32)
                labels=raw[:,-1].astype(int)
                boundaries=np.r_[0,np.flatnonzero(np.diff(labels))+1,len(labels)]
                for begin,end in zip(boundaries[:-1],boundaries[1:]):
                    label=int(labels[begin])
                    if label==0: continue
                    for offset in range(begin,end-100+1,50):
                        segment=raw[offset:offset+100]
                        windows.append(np.stack([segment[:,ids] for ids in [[0,1,2],[14,15,16],[5,6,7]]]))
                        targets.append(label-1);people.append(int(match.group(1)))
        x=np.stack(windows)
        candidates=['standing','sitting','lying','walking','climbing stairs','bending waist forward','raising arms in front','bending knees','cycling','jogging','running','jumping front and back']
        yield {'dataset':'mhealth','x':adapt(x,50),'y':np.asarray(targets),'subjects':np.asarray(people),
               'classes':candidates,'candidates':candidates,'positions':['chest','right_lower_arm','left_ankle'],
               'source':str(artifacts/'mhealth.zip'),'source_shape':list(x.shape),'sample_rate_hz':50,
               'selection':'all 10 subjects; 100-step windows / stride 50 within contiguous activity blocks; null excluded; no retraining'}
    if (artifacts/'dsads.zip').exists():
        windows=[];targets=[];people=[]
        with zipfile.ZipFile(artifacts/'dsads.zip') as archive:
            for member in sorted(archive.namelist()):
                match=re.fullmatch(r'data/a(\d+)/p(\d+)/s\d+\.txt',member)
                if not match:continue
                with archive.open(member) as stream:raw=np.loadtxt(stream,delimiter=',',dtype=np.float32)
                assert raw.shape==(125,45)
                windows.append(np.stack([raw[:,i:i+3] for i in [0,9,18,27,36]]))
                targets.append(int(match.group(1))-1);people.append(int(match.group(2)))
        x=np.stack(windows)
        candidates=['sitting','standing','lying on back','lying on right side','upstairs','downstairs','standing still in elevator','moving around in elevator','walking in parking lot','walking on level treadmill','walking on inclined treadmill','running on treadmill','exercising on stepper','exercising on cross trainer','cycling in horizontal position','cycling in vertical position','rowing','jumping','playing basketball']
        yield {'dataset':'dsads','x':adapt(x,25),'y':np.asarray(targets),'subjects':np.asarray(people),
               'classes':candidates,'candidates':candidates,'positions':['torso','right_arm','left_arm','right_leg','left_leg'],
               'source':str(artifacts/'dsads.zip'),'source_shape':list(x.shape),'sample_rate_hz':25,
               'selection':'all original 9120 five-second segments; 19 actual source classes; no custom class merge; no retraining'}

    if (artifacts/'unimib_shar.zip').exists():
        with zipfile.ZipFile(artifacts/'unimib_shar.zip') as archive:
            raw=loadmat(io.BytesIO(archive.read('UniMiB-SHAR/data/acc_data.mat')))['acc_data']
            labels=loadmat(io.BytesIO(archive.read('UniMiB-SHAR/data/acc_labels.mat')))['acc_labels']
        assert raw.shape==(11771,453) and labels.shape==(11771,3)
        x=raw.reshape(-1,3,151).transpose(0,2,1)[:,None]
        candidates=['standing up from sitting','standing up from lying','walking','running','upstairs','jumping','downstairs','lying down from standing','sitting down','falling forward','falling right','falling backward','hitting an obstacle','falling with protection','falling backward from chair','fainting','falling left']
        yield {'dataset':'unimib_shar','x':adapt(x,50),'y':labels[:,0].astype(int)-1,'subjects':labels[:,1].astype(int),
               'classes':candidates,'candidates':candidates,'positions':['phone pocket'],
               'source':str(artifacts/'unimib_shar.zip'),'source_shape':list(x.shape),'sample_rate_hz':50,
               'selection':'all 11771 original AF-17 windows / 30 subjects; 17 source classes rather than placeholder class groups; no retraining'}
    if (artifacts/'skoda.zip').exists():
        with zipfile.ZipFile(artifacts/'skoda.zip') as archive:
            raw=loadmat(io.BytesIO(archive.read('SkodaMiniCP_2015_08/right_classall_clean.mat')))['right_classall_clean']
        assert raw.ndim==2 and raw.shape[1]==71
        labels=raw[:,0].astype(int)
        boundaries=np.r_[0,np.flatnonzero(np.diff(labels))+1,len(labels)]
        windows=[]; targets=[]
        for begin,end in zip(boundaries[:-1],boundaries[1:]):
            label=int(labels[begin])
            if label not in range(48,58): continue
            for offset in range(begin,end-30+1,15):
                segment=raw[offset:offset+30]
                windows.append(np.stack([segment[:,2+7*i:5+7*i] for i in range(10)]))
                targets.append(label-48)
        x=np.stack(windows)
        candidates=['writing on notepad','opening car hood','closing car hood','checking gaps on front door','opening left front car door','closing left front car door','closing both left car doors','checking trunk gaps','opening and closing trunk','checking steering wheel']
        yield {'dataset':'skoda','x':adapt(x,98),'y':np.asarray(targets),'subjects':np.ones(len(targets),dtype=int),
               'classes':candidates,'candidates':candidates,'positions':[f'right_arm_accelerometer_{i+1}' for i in range(10)],
               'source':str(artifacts/'skoda.zip'),'source_shape':list(x.shape),'sample_rate_hz':98,
               'selection':'single subject; calibrated acceleration from all 10 right-arm sensors; 30-step windows / stride 15 inside gestures; null excluded; no retraining',
               'sensor_fusion':'two groups of five sensors; average cosine-similarity scores; required by released five-position ViT maximum'}


def run(model,args,module,output,label_forward,load_audit,cache_audit,options):
    root=Path(__file__).resolve().parent.parent
    device=next(model.parameters()).device
    results=[]
    start=time.time()
    selected=set(options.datasets.split(',')) if options.datasets else None
    for case in available_cases(root,options):
        if selected and case['dataset'] not in selected: continue
        name=case['dataset']; x=case.pop('x'); y=case.pop('y'); subjects=case.pop('subjects')
        assert x.shape[0]==len(y)==len(subjects) and len(case['classes'])==len(case['candidates'])
        assert set(y.tolist()) <= set(range(len(case['classes'])))
        model.label_encoder.label_map=case['candidates']
        with torch.no_grad():
            vectors=torch.nn.functional.normalize(label_forward(torch.arange(len(case['candidates']),device=device)),dim=-1)
            predictions=[];scores=[]
            for i in range(0,len(x),256):
                imu=torch.from_numpy(x[i:i+256]).to(device)
                group_scores=[]
                for sensor_begin in range(0,imu.shape[1],5):
                    group=imu[:,sensor_begin:sensor_begin+5].contiguous()
                    feature=torch.nn.functional.normalize(model.imu_encoder(group,args),dim=-1)
                    group_scores.append(feature@vectors.T)
                logits=torch.stack(group_scores).mean(dim=0).cpu().numpy()
                predictions.append(logits.argmax(axis=1));scores.append(logits)
        pred=np.concatenate(predictions)
        by_subject=[]
        for subject in np.unique(subjects):
            mask=subjects==subject
            by_subject.append({'subject':int(subject),'samples':int(mask.sum()),'accuracy':accuracy_score(y[mask],pred[mask]),'f1_macro':f1_score(y[mask],pred[mask],labels=list(range(len(case['classes']))),average='macro',zero_division=0)})
        result={**case,'samples':len(y),'num_classes':len(case['classes']),'accuracy':accuracy_score(y,pred),
                'f1_macro':f1_score(y,pred,labels=list(range(len(case['classes']))),average='macro',zero_division=0),
                'f1_weighted':f1_score(y,pred,average='weighted',zero_division=0),
                'confusion_matrix':confusion_matrix(y,pred,labels=list(range(len(case['classes'])))).tolist(),
                'by_subject':by_subject,'subject_mean_accuracy':float(np.mean([a['accuracy'] for a in by_subject])),
                'subject_std_accuracy':float(np.std([a['accuracy'] for a in by_subject]))}
        np.savez_compressed(output/f'{name}_predictions.npz',true=y,pred=pred,subjects=subjects,logits=np.concatenate(scores))
        results.append(result)
        report={'suite':'exploratory frozen-checkpoint input/label adapters','not_canonical_leaderboard_protocol':True,
                'protocol':'accelerometer only; physical-time center crop/edge pad to 2.4 s and resample to 50 Hz; test-position scalar normalization; batch size 256; dataset-specific textual class candidates; >5 sensors grouped in fives and scores averaged; no training',
                'training_provenance':'Released checkpoint training membership is not supplied; do not claim unseen subjects/datasets or new LOSO training.',
                'checkpoint_audit':load_audit,'cache_audit':cache_audit,'results':results,'elapsed_seconds':time.time()-start}
        (output/'results.json').write_text(json.dumps(report,indent=2))
        print('ADAPTED_RESULT',json.dumps({k:result[k] for k in ['dataset','samples','num_classes','accuracy','f1_macro']}),flush=True)
    with (output/'summary.csv').open('w',newline='') as f:
        fields=['dataset','samples','num_classes','accuracy','f1_macro','f1_weighted','subject_mean_accuracy','subject_std_accuracy']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
