"""Author phone/wearable IF-ConvTransformer; explicit physical sensor adapter."""
import importlib.util
import json
import sys
import types
import numpy as np
from scipy.signal import butter,sosfiltfilt
import torch
from torch import nn

from prepare import HERE,sha


def adapt_raw(x,dataset):
    if dataset=='uci_har':return np.concatenate([x[:,:,6:9]-x[:,:,:3],x[:,:,3:6],x[:,:,:3]],2).astype(np.float32)
    rate={'wisdm':20,'hhar':50,'pamap2':100,'opportunity_a':30}[dataset]
    sensors=[]
    if dataset in ['hhar','wisdm']:
        sensors=[(x[:,:,:3],x[:,:,3:6] if dataset=='hhar' else np.zeros_like(x))]
    elif dataset=='pamap2':sensors=[(x[:,:,i:i+3],x[:,:,i+3:i+6]) for i in [0,9,18]]
    else:
        sensors=[(x[:,:,i:i+3],np.zeros_like(x[:,:,i:i+3])) for i in range(0,36,3)]
        sensors += [(x[:,:,i:i+3],x[:,:,i+3:i+6]) for i in [36,45,54,63,72]]
        sensors += [(x[:,:,i+6:i+9],x[:,:,i+9:i+12]) for i in [81,97]]
    sos=butter(3,.3,fs=rate,output='sos')
    values=[]
    for acc,gyro in sensors:
        gravity=sosfiltfilt(sos,acc,axis=1).astype(np.float32)
        values.append(np.concatenate([gravity,gyro,acc-gravity],2))
    return np.concatenate(values,2).astype(np.float32)


class LabelSmoothingCrossEntropy(nn.Module):
    def forward(self,output,target):
        log=torch.log_softmax(output,-1)
        return .1*(-log.mean(-1)).mean()+.9*nn.functional.nll_loss(log,target)


def build(dataset,length,channels,classes):
    # Shims cover logging/optional training imports only; model arithmetic is
    # the unchanged author source. The source's train_op is not invoked.
    sys.modules['torchvision']=types.ModuleType('torchvision')
    plot=types.ModuleType('matplotlib.pyplot');mat=types.ModuleType('matplotlib');mat.pyplot=plot
    sys.modules['matplotlib']=mat;sys.modules['matplotlib.pyplot']=plot
    utils=types.ModuleType('utils');u=types.ModuleType('utils.utils')
    u.LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy
    sys.modules['utils']=utils;sys.modules['utils.utils']=u
    contiguous=types.ModuleType('contiguous_params')
    contiguous.ContiguousParams=lambda p:p
    sys.modules['contiguous_params']=contiguous
    wearable=dataset in ['pamap2','opportunity_a']
    name='If_ConvTransformer_W' if wearable else 'If_ConvTransformer'
    source=HERE/'source/IF-ConvTransformer-UbiComp2022/src/classifiers'/f'{name}_torch.py'
    spec=importlib.util.spec_from_file_location('official_if_model',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    model=getattr(m,name)(1,channels,64,5,3,2,128,1,.2,'Standard',length,classes)
    if wearable:
        model.apply(m.init_weights_orthogonal)
        model._loss_ops=m
    return model,{'optimizer':'Adam','lr':.001,'batch_size':8 if dataset=='opportunity_a' else 64,'weight_decay':0.,
        'loss':'author mixup + center loss' if wearable else 'label_smoothing_0.1',
        'architecture':f'unchanged author {name}, F64, K5, gravity K3, scales2, output128, heads1, dropout .2',
        'official_repo':'https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022',
        'official_commit':'577fa56ae74978276030372d84b17ee0f1d000ce','implementation_type':'official_code_physical_sensor_adapter',
        'model_sha256':sha(source),'adapter_sha256':sha(__file__),'input':'B,1,C,T','arithmetic':'float32',
        'logits_index':0,'use_mixup':wearable,'beta':.0003,'center_lr':.001,
        'sensor_adapter':'UCI exact total-minus-body gravity; other targets window-local 3rd-order 0.3-Hz zero-phase Butterworth gravity + gyro + body acceleration. Missing gyro zero. Magnetometers and extra shoe orientation/nav channels excluded; all 19 applicable OPPORTUNITY sensor streams retained.',
        'calibration':'physical sensor adapter differs from original paper loader; calibration pending. Old first-nine channel truncations are excluded.'}
