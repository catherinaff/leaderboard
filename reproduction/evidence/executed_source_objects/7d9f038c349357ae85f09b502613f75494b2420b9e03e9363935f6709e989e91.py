"""Author HART/MobileHART/iSPLInception code on the frozen continuation splits.

This is a canonical-data adaptation, not an exact-paper-protocol reproduction.
The first physical acc/gyro pair is retained, as required by the author models.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from prepare import HERE,ROOT,atomic_json,sha

REPO=ROOT/'experiments/repos/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices'


def metrics(y,p,classes):
    return dict(accuracy=float(accuracy_score(y,p)),
        f1_macro=float(f1_score(y,p,labels=list(range(classes)),average='macro',zero_division=0)),
        f1_weighted=float(f1_score(y,p,labels=list(range(classes)),average='weighted',zero_division=0)))


def run(a):
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    # The launcher sets LD_LIBRARY_PATH before exec so the dynamic loader can
    # resolve TF's CUDA dependencies without changing either conda environment.
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    gpus=tf.config.list_physical_devices('GPU')
    if not gpus:raise RuntimeError('TensorFlow GPU libraries unavailable; refusing to silently switch the formal queue to CPU')
    total=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=int(total*a.memory_fraction))])
    random.seed(a.seed);np.random.seed(a.seed);tf.keras.utils.set_random_seed(a.seed)
    started=time.time()
    out=HERE/('smoke' if a.smoke else 'runs')/a.method/a.dataset/f'seed_{a.seed}'/a.fold
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'result.json').exists():print('already complete',out,flush=True);return
    archive=out/'execution_source';archive.mkdir(exist_ok=True)
    sources={p.name:p.read_bytes() for p in HERE.glob('*.py')}
    ispl=a.method=='ispl_inception'
    repo=HERE/'vendor/ispl-source' if ispl else REPO
    model_file=repo/('models.py' if ispl else 'model.py')
    sources['official_model.py']=model_file.read_bytes()
    committed=subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+model_file.name])
    assert committed==model_file.read_bytes(),'Author model code differs from its recorded commit'
    for name,content in sources.items():(archive/name).write_bytes(content)
    source=HERE/'data'/a.dataset
    # HART's two branches consume one 3-axis accelerometer/gyroscope pair.
    # UCI: body acc/gyro. PAMAP2: hand. RealWorld: chest. USC-HAD: native IMU.
    raw=np.load(source/'x.npy',mmap_mode='r')
    selected=np.arange(raw.shape[2] if ispl else 6)
    x=raw[:,:,selected].copy();y=np.load(source/'y.npy')
    subjects=np.load(source/'subjects.npy')
    channels=x.shape[2]
    assert ispl or channels==6
    split=HERE/'manifests/splits'/a.dataset/f'seed_{a.seed}'/f'{a.fold}.npz'
    indices=np.load(split);tr,va,te=[indices[k] for k in ['train','validation','test']]
    if a.smoke:
        rng=np.random.default_rng(a.seed)
        def sample(ids,n):return np.concatenate([rng.permutation(ids[y[ids]==c])[:n] for c in np.unique(y[ids])])
        tr,va,te=sample(tr,6),sample(va,2),sample(te,2)
    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
    median=np.nanmedian(x[tr].reshape(-1,channels),axis=0);assert np.isfinite(median).all()
    x=np.where(np.isfinite(x),x,median)
    mean=x[tr].mean((0,1),dtype=np.float64).astype(np.float32)
    std=x[tr].std((0,1),dtype=np.float64).astype(np.float32).clip(1e-6)
    x=((x-mean)/std).astype(np.float32);assert np.isfinite(x).all()
    np.savez_compressed(out/'scaler.npz',median=median,mean=mean,std=std,channel_indices=selected)
    np.savez_compressed(out/'split.npz',train=tr,validation=va,test=te)
    bench=json.loads((ROOT/'benchmarks'/f'{a.dataset}.json').read_text())
    primary=bench['metrics']['primary'];classes=len(bench['classes'])
    spec=importlib.util.spec_from_file_location('official_tf_model',model_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if ispl:
        model=module.ispl_inception(x.shape,classes,filters_number=64,network_depth=5,
            use_residual=True,use_bottleneck=True,max_kernel_size=68,learning_rate=.0005,regularization_rate=.00593)
    else:
        build=module.HART if a.method=='hart' else module.mobileHART_XS
        model=build((x.shape[1],channels),classes)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    batch=a.batch_size or (64 if ispl else 256)
    lr=.0005 if ispl else .005
    recipe=dict(optimizer='Adam AMSGrad' if ispl else 'Adam',lr=lr,batch_size=batch,weight_decay=0.,
        loss='author categorical cross entropy + layer L2 regularizers' if ispl else 'author categorical cross entropy, label smoothing .1, balanced training class weights',
        architecture='author iSPLInception, 5 blocks, F64, kernels 68/34/17, bottleneck32, residual, L2 .00593' if ispl else 'author HART defaults' if a.method=='hart' else 'author mobileHART_XS defaults',
        official_repo='https://github.com/rmutegeki/iSPLInception' if ispl else 'https://github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices',
        official_commit=commit,model_sha256=sha(model_file),runner_sha256=sha(__file__),
        implementation_type='official_code_canonical_data_adapter',input=f'B,T,{channels}',
        sensor_selection='all native canonical channels' if ispl else dict(uci_har='body_acc + body_gyro',pamap2='hand acc + gyro',realworld_har='chest acc + gyro',usc_had='native acc + gyro')[a.dataset],
        adapter=('All native channels' if ispl else 'First physical acc/gyro pair')+'; native canonical window length; original model file is unchanged.',
        protocol_difference='Canonical subject splits, train-only scaling and common validation early stopping differ from the paper protocol. Requires independent calibration.',
        max_epochs=a.epochs,patience=25,min_delta=.0005,monitor=primary,
        scheduler='ReduceLROnPlateau factor .5 patience 8 min_lr 1e-6',
        gradient_clip=1.,scaler_fit='train_windows_only',
        validation_subjects=sorted(set(map(int,subjects[va]))),test_subjects=sorted(set(map(int,subjects[te]))))
    counts=np.bincount(y[tr],minlength=classes)
    weights=np.divide(len(tr),classes*counts,out=np.zeros(classes,np.float64),where=counts>0).astype(np.float32)
    if ispl:weights=np.ones(classes,np.float32)
    loss_fn=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0. if ispl else .1,reduction='none')
    optimizer=tf.keras.optimizers.Adam(lr,amsgrad=ispl)
    @tf.function(reduce_retracing=True)
    def step(bx,by,bw):
        with tf.GradientTape() as tape:
            probs=model(bx,training=True)
            loss=tf.reduce_mean(loss_fn(tf.one_hot(by,classes),probs)*bw)
            if model.losses:loss+=tf.add_n(model.losses)
            tf.debugging.assert_all_finite(loss,'non-finite training loss')
        gradients=tape.gradient(loss,model.trainable_variables)
        pairs=[(g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None]
        for g,_ in pairs:tf.debugging.assert_all_finite(g,'non-finite gradient')
        clipped,_=tf.clip_by_global_norm([g for g,_ in pairs],1.)
        optimizer.apply_gradients(zip(clipped,[v for _,v in pairs]))
        return loss
    @tf.function(reduce_retracing=True)
    def forward(bx):return model(bx,training=False)
    def predict(ids):
        values=np.concatenate([forward(x[ids[i:i+batch]]).numpy() for i in range(0,len(ids),batch)])
        assert np.isfinite(values).all()
        return values
    meta=dict(method=a.method,dataset=a.dataset,seed=a.seed,fold=a.fold,recipe=recipe,
        run_mode='smoke' if a.smoke else 'canonical_candidate',train_class_counts=counts.tolist(),
        dataset_manifest_sha256=sha(source/'manifest.json'),split_sha256=sha(split),runner_sha256=sha(__file__),
        execution_source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},
        device=tf.config.experimental.get_device_details(gpus[0]).get('device_name','GPU'),tensorflow=tf.__version__,
        normalization=f'training median then training mean/std; {channels} channels',
        counts=dict(train=len(tr),validation=len(va),test=len(te)),status='running')
    atomic_json(out/'run.json',meta)
    history=[];best=-1.;best_epoch=0;stale=0;lr_stale=0
    for epoch in range(1,a.epochs+1):
        tick=time.time();order=np.random.permutation(tr);total_loss=0.
        for i in range(0,len(order),batch):
            ids=order[i:i+batch]
            total_loss+=float(step(x[ids],y[ids],weights[y[ids]]).numpy())*len(ids)
        validation=metrics(y[va],predict(va).argmax(1),classes)
        score=validation[primary]
        if score>best+.0005:
            best=score;best_epoch=epoch;stale=lr_stale=0
            temp=out/'best.tmp.weights.h5';model.save_weights(str(temp));temp.replace(out/'best.weights.h5')
        else:
            stale+=1;lr_stale+=1
            if lr_stale>=8:
                optimizer.learning_rate.assign(max(1e-6,float(optimizer.learning_rate.numpy())*.5));lr_stale=0
        history.append(dict(epoch=epoch,train_loss=total_loss/len(tr),validation=validation,
            lr=float(optimizer.learning_rate.numpy()),seconds=time.time()-tick))
        atomic_json(out/'history.json',history)
        if epoch==1 or epoch%5==0 or stale>=25:print(a.method,a.dataset,a.seed,a.fold,history[-1],flush=True)
        if stale>=25:break
    model.load_weights(str(out/'best.weights.h5'))
    vp=predict(va);tp=predict(te)
    np.savez_compressed(out/'predictions.npz',test_indices=te,y_true=y[te],probabilities=tp,y_pred=tp.argmax(1),
        validation_indices=va,validation_probabilities=vp,validation_y_true=y[va])
    result=dict(**meta);result.update(status='smoke_complete' if a.smoke else 'complete',epochs_run=epoch,best_epoch=best_epoch,
        validation=metrics(y[va],vp.argmax(1),classes),test=metrics(y[te],tp.argmax(1),classes),
        confusion_matrix=confusion_matrix(y[te],tp.argmax(1),labels=list(range(classes))).tolist(),
        runtime_seconds=time.time()-started,completed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        artifacts={name:sha(out/name) for name in ['best.weights.h5','predictions.npz','scaler.npz','split.npz']})
    atomic_json(out/'result.json',result);print('COMPLETE',result['test'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['hart','mobilehart','ispl_inception'],required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int);p.add_argument('--memory-fraction',type=float,default=.16);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke and args.epochs!=200:raise ValueError('Formal queue fixes the budget at 200 epochs')
    run(args)
