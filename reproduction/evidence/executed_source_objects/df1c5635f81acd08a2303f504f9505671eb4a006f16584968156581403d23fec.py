"""Resumable, artifact-producing Standard runner. Paper calibration is separate."""
import argparse
import copy
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import nn

from prepare import HERE, ROOT, SEEDS, atomic_json, sha

SOURCE_BYTES={p.name:p.read_bytes() for p in HERE.iterdir() if p.suffix in {'.py','.md'}}


def metrics(y,p,classes):
    return {'accuracy':float(accuracy_score(y,p)),
            'f1_macro':float(f1_score(y,p,labels=list(range(classes)),average='macro',zero_division=0)),
            'f1_weighted':float(f1_score(y,p,labels=list(range(classes)),average='weighted',zero_division=0))}


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def build(method,length,channels,classes,**context):
    if method=='tinyhar':
        src=HERE/'source/ISWC22-HAR/models/TinyHAR.py'
        m=module(src,'official_tinyhar')
        net=m.TinyHAR_Model((1,1,length,channels),classes,filter_num=20,
              cross_channel_interaction_type='attn',cross_channel_aggregation_type='FC',
              temporal_info_interaction_type='lstm',temporal_info_aggregation_type='tnaive')
        recipe={'optimizer':'Adam','lr':.001,'batch_size':128,'weight_decay':0.,
                'loss':'cross_entropy','architecture':'author TinyHAR_Model, F=20, attn/FC/lstm/tnaive',
                'official_repo':'https://github.com/teco-kit/ISWC22-HAR',
                'official_commit':'b84b89d09f6914fe93e82cde423e294042da2741',
                'implementation_type':'official_code','model_sha256':sha(src),'input':'B,1,T,C',
                'arithmetic':'float32; architecture and operations unchanged'}
        return net,recipe
    if method=='attend_and_discriminate':
        src=HERE/'source/Attend-And-Discriminate'
        sys.path.insert(0,str(src))
        m=module(src/'models.py','official_ad')
        pamap=channels==27
        net=m.AttendDiscriminate('AttendDiscriminate','standard',channels,128,64,5,2,False,
                                .9 if pamap else .5,0. if pamap else .25,.5,'ReLU',1,
                                classes,True,'test_models')
        if not pamap:
            from utils.utils_pytorch import init_weights_orthogonal
            net.apply(init_weights_orthogonal)
        recipe={'optimizer':'Adam','lr':.001,'batch_size':128,'weight_decay':0.,
                'loss':'author MixUpLoss(alpha=.8) + center loss',
                'beta':.003 if pamap else .0003,'center_lr':.001,
                'source_recipe':'PAMAP2' if pamap else 'OPPORTUNITY (closest available author recipe)',
                'architecture':'author AttendDiscriminate, 4Conv64 + 2GRU128 + sensor/temporal attention',
                'official_repo':'https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate',
                'official_commit':json.loads((src/'SOURCE.json').read_text())['commit'],
                'implementation_type':'official_code','model_sha256':sha(src/'models.py'),'input':'B,T,C',
                'arithmetic':'float32'}
        return net,recipe
    if method=='robusthar':
        from robusthar import prepare_model
        return prepare_model(context['x'],context['train'],context['validation'],context['out'],context['a'],context['device'])
    if method=='limu_bert':
        from limu import prepare_model
        return prepare_model(context['x'],context['train'],context['validation'],context['out'],context['a'],context['device'])
    if method=='deepconvlstm':
        from deepconvlstm import DeepConvLSTM
        return DeepConvLSTM(channels,classes),{
            'optimizer':'RMSprop','lr':.001,'batch_size':100,'weight_decay':0.,'loss':'cross_entropy',
            'architecture':'paper/Lasagne port: 4 valid 5x1 Conv64, 2 peephole LSTM128, dense dropout .5',
            'official_repo':'https://github.com/STRCWearlab/DeepConvLSTM',
            'official_commit':'ea25e34da3a48b151a97bd3502802db547523b7f',
            'implementation_type':'paper_reimplementation_with_source_audit','model_sha256':sha(HERE/'deepconvlstm.py'),
            'input':'B,1,T,C','arithmetic':'float32','calibration':'pending checkpoint equivalence and paper-protocol training audit',
            'source_note':'Original paper Table1 contains peephole weights; the older Conv1d/ordinary-LSTM proxy is excluded. Standard adapts only input channels/output classes and window length.'}
    if method=='deepsense':
        from deepsense import DeepSense
        dataset=context['a'].dataset
        net=DeepSense(dataset,channels,classes)
        return net,{
            'optimizer':'Adam','lr':.0001,'batch_size':64,'weight_decay':.0005,'loss':'cross_entropy',
            'architecture':'author TF1 operation port: FFT + modality CNNs + fusion CNNs + 2 reset-before GRU120',
            'official_repo':'https://github.com/yscacaca/DeepSense',
            'official_commit':'1979ba414d3cfd0a84a5247e2713540e8c225ec7',
            'implementation_type':'pytorch_port_of_author_tf1_code','model_sha256':sha(HERE/'deepsense.py'),
            'input':'B,T,C','arithmetic':'float32','sensor_groups':net.groups,
            'adapter':'10-native-sample FFT subwindows, zero-pad final partial subwindow; all native channels/modalities retained. This differs from the original 5-second/10-spectral-sample HHAR preprocessing and requires separate calibration.',
            'calibration':'pending paper-protocol and numerical TF1/PyTorch comparison; BN running-variance estimators differ between frameworks'}
    if method=='unimts':
        from unimts_adapter import prepare_model
        return prepare_model(context['x'],context['train'],context['validation'],context['out'],context['a'],context['device'])
    if method=='if_convtransformer':
        from if_adapter import build as build_if
        return build_if(context['a'].dataset,length,channels,classes)
    raise ValueError(f'No audited adapter for {method}')


def run(a):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.cuda.set_device(a.gpu)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,a.gpu)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device(f'cuda:{a.gpu}')
    out=HERE/('smoke' if a.smoke else 'runs')/a.method/a.dataset/f'seed_{a.seed}'/a.fold
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'result.json').exists():
        print('already complete',out,flush=True);return
    archive=out/'execution_source';archive.mkdir(exist_ok=True)
    for name,content in SOURCE_BYTES.items():(archive/name).write_bytes(content)
    started=time.time()
    source=HERE/'data'/a.dataset
    x=np.load(source/'x.npy').copy();y=np.load(source/'y.npy')
    subjects=np.load(source/'subjects.npy')
    split=HERE/'manifests/splits'/a.dataset/f'seed_{a.seed}'/f'{a.fold}.npz'
    indices=np.load(split);tr,va,te=[indices[k] for k in ['train','validation','test']]
    if a.smoke:
        rng=np.random.default_rng(a.seed)
        def subset(ids,n):
            return np.concatenate([rng.permutation(ids[y[ids]==c])[:n] for c in np.unique(y[ids])])
        tr,va,te=subset(tr,6),subset(va,2),subset(te,2)
    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
    # Median and z-score are fitted exclusively on training windows.
    median=np.nanmedian(x[tr].reshape(-1,x.shape[-1]),axis=0)
    if not np.isfinite(median).all():
        raise ValueError('A channel has no finite training samples; protocol requires a documented decision')
    missing=~np.isfinite(x)
    x[missing]=np.broadcast_to(median,x.shape)[missing]
    if a.method=='if_convtransformer':
        from if_adapter import adapt_raw
        x=adapt_raw(x,a.dataset)
    mean=x[tr].mean((0,1),dtype=np.float64).astype(np.float32)
    std=x[tr].std((0,1),dtype=np.float64).astype(np.float32).clip(1e-6)
    x=(x-mean)/std
    assert np.isfinite(x).all()
    np.savez_compressed(out/'scaler.npz',median=median,mean=mean,std=std)
    np.savez_compressed(out/'split.npz',train=tr,validation=va,test=te)
    bench=json.loads((ROOT/'benchmarks'/f'{a.dataset}.json').read_text())
    primary=bench['metrics']['primary'];classes=len(bench['classes'])
    model,recipe=build(a.method,x.shape[1],x.shape[2],classes,x=x,train=tr,validation=va,out=out,a=a,device=device)
    if '_input_array' in recipe:x=recipe.pop('_input_array')
    recipe.update({'max_epochs':min(a.epochs,recipe.get('max_epochs',a.epochs)),'patience':25,'min_delta':.0005,
                   'monitor':primary,'scheduler':'ReduceLROnPlateau factor=.5 patience=8 min_lr=1e-6',
                   'gradient_clip':1.,'scaler_fit':'train_windows_only',
                   'validation_subjects':sorted(set(map(int,subjects[va]))),
                   'test_subjects':sorted(set(map(int,subjects[te])))})
    model=model.float().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'])
    if recipe['optimizer']=='RMSprop':
        from deepconvlstm import LasagneRMSprop
        opt=LasagneRMSprop(model.parameters(),lr=recipe['lr'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=8,min_lr=1e-6)
    counts=np.bincount(y[tr],minlength=classes)
    # TinyHAR official recipe specifies ordinary CE, so retain it regardless of imbalance.
    loss_fn=nn.CrossEntropyLoss()
    mixup_enabled=a.method=='attend_and_discriminate' or recipe.get('use_mixup',False)
    if a.method=='attend_and_discriminate':
        from utils.utils_mixup import MixUpLoss,mixup_data
        from utils.utils_centerloss import compute_center_loss,get_center_delta
    elif mixup_enabled:
        hooks=model._loss_ops
        MixUpLoss,mixup_data=hooks.MixUpLoss,hooks.mixup_data
        compute_center_loss,get_center_delta=hooks.compute_center_loss,hooks.get_center_delta
    elif recipe['loss']=='label_smoothing_0.1':
        from if_adapter import LabelSmoothingCrossEntropy
        loss_fn=LabelSmoothingCrossEntropy()
    if mixup_enabled:
        loss_fn=MixUpLoss(nn.CrossEntropyLoss())
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.9)
        recipe['scheduler']='author StepLR(step_size=10,gamma=.9)'
    gx=torch.as_tensor(x,device=device)
    if recipe['input']=='B,1,T,C':gx=gx.unsqueeze(1)
    elif recipe['input']=='B,1,C,T':gx=gx.transpose(1,2).unsqueeze(1)
    gy=torch.as_tensor(y,device=device)
    batch=a.batch_size or recipe['batch_size']
    recipe['batch_size']=batch
    meta={'method':a.method,'dataset':a.dataset,'seed':a.seed,'fold':a.fold,'recipe':recipe,
          'run_mode':'smoke' if a.smoke else 'canonical_candidate',
          'train_class_counts':counts.tolist(),'dataset_manifest_sha256':sha(source/'manifest.json'),
          'split_sha256':sha(split),'runner_sha256':hashlib.sha256(SOURCE_BYTES['train.py']).hexdigest(),
          'execution_source_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in SOURCE_BYTES.items()},
          'device':torch.cuda.get_device_name(device),
          'torch':torch.__version__,'normalization':'training median then training mean/std',
          'counts':{'train':len(tr),'validation':len(va),'test':len(te)},'status':'running'}
    atomic_json(out/'run.json',meta)
    def predict(ids):
        model.eval();logits=[]
        with torch.no_grad():
            for i in range(0,len(ids),batch):
                o=model(gx[ids[i:i+batch]])
                if isinstance(o,tuple):o=o[recipe.get('logits_index',-1)]
                if o.ndim==1:o=o.unsqueeze(0)
                logits.append(o.detach().cpu().numpy())
        return np.concatenate(logits)
    best=-1.;stale=0;history=[];best_epoch=0
    for epoch in range(1,recipe['max_epochs']+1):
        t=time.time();model.train();order=np.random.permutation(tr);loss_sum=0.;n=0
        for i in range(0,len(order),batch):
            ids=order[i:i+batch]
            # BatchNorm based official models cannot train on a singleton.
            if len(ids)==1 and i>0:
                ids=order[i-1:i+1]
            opt.zero_grad(set_to_none=True)
            if mixup_enabled:
                mixed,targets=mixup_data(gx[ids],gy[ids],.8)
                pair=model(mixed)
                if recipe.get('logits_index',-1)==0:output,z=pair
                else:z,output=pair
                loss=loss_fn(output,targets)+recipe['beta']*compute_center_loss(z,model.centers,gy[ids])
            else:
                output=model(gx[ids])
                if isinstance(output,tuple):output=output[recipe.get('logits_index',-1)]
                loss=loss_fn(output,gy[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if mixup_enabled:
                model.centers=model.centers-get_center_delta(z.detach(),model.centers,gy[ids],recipe['center_lr'])
            loss_sum+=float(loss.detach())*len(ids);n+=len(ids)
        vp=predict(va).argmax(1);v=metrics(y[va],vp,classes)
        if mixup_enabled:scheduler.step()
        else:scheduler.step(v[primary])
        row={'epoch':epoch,'train_loss':loss_sum/n,'validation':v,'lr':opt.param_groups[0]['lr'],
             'seconds':time.time()-t}
        history.append(row)
        if v[primary]>best+.0005:
            best=v[primary];best_epoch=epoch;stale=0
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'validation':v,
                        'recipe':recipe},out/'best.pt.tmp')
            (out/'best.pt.tmp').replace(out/'best.pt')
        else:stale+=1
        atomic_json(out/'history.json',history)
        if epoch==1 or epoch%5==0 or stale>=25:
            print(a.method,a.dataset,a.seed,a.fold,epoch,round(loss_sum/n,5),v,round(time.time()-t,2),flush=True)
        if stale>=25:break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    vl=predict(va);tl=predict(te)
    np.savez_compressed(out/'predictions.npz',test_indices=te,y_true=y[te],logits=tl,y_pred=tl.argmax(1),
                        validation_indices=va,validation_logits=vl,validation_y_true=y[va])
    result={**meta,'status':'smoke_complete' if a.smoke else 'complete','epochs_run':epoch,'best_epoch':best_epoch,
            'validation':metrics(y[va],vl.argmax(1),classes),'test':metrics(y[te],tl.argmax(1),classes),
            'confusion_matrix':confusion_matrix(y[te],tl.argmax(1),labels=list(range(classes))).tolist(),
            'runtime_seconds':time.time()-started,'completed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'artifacts':{p.name:sha(p) for p in [out/'best.pt',out/'predictions.npz',out/'scaler.npz',out/'split.npz']}}
    atomic_json(out/'result.json',result)
    print('COMPLETE',result['test'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',required=True)
    p.add_argument('--gpu',type=int,default=3);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--pretrain-batch-size',type=int)
    p.add_argument('--memory-fraction',type=float,default=.18)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke and args.epochs!=200:raise ValueError('Production queue fixes the maximum budget at 200 epochs; use --smoke for short validation runs')
    run(args)
