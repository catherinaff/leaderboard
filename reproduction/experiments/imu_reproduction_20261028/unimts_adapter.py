"""Official UniMTS ST-GCN full fine-tuning, physical sensor input adapter.

Uses the author checkpoint's IMU branch, not unused CLIP text/vision modules.
Source model and fine-tune operations are preserved. Sensor subsets are explicit.
"""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import resample

from prepare import HERE,sha


def adapt(x,dataset):
    if dataset=='uci_har':
        values=np.concatenate([x[:,:,6:9],x[:,:,3:6]],2);joints=[9];rate=50
        note='total acceleration + gyroscope; body-acceleration duplicate excluded; waist joint 9'
    elif dataset=='wisdm':
        values=np.concatenate([x,np.zeros_like(x)],2);joints=[9];rate=20
        note='AR v1.1 phone-in-pocket accelerometer at waist joint 9; missing gyroscope is zero'
    elif dataset=='hhar':
        values=x;joints=[9];rate=50;note='phone acceleration/gyroscope at waist joint 9'
    elif dataset=='pamap2':
        values=np.concatenate([x[:,:,i:i+6] for i in [0,9,18]],2);joints=[21,11,7];rate=100
        note='hand/chest/ankle acceleration+gyro at author joints 21/11/7; magnetometers excluded'
    elif dataset=='opportunity_a':
        values=np.concatenate([x[:,:,i:i+6] for i in [36,45,54,63,72]]+
                              [np.concatenate([x[:,:,i+6:i+9],x[:,:,i+9:i+12]],2) for i in [81,97]],2)
        joints=[11,19,20,15,16,8,4];rate=30
        note='5 body IMUs + 2 shoe IMUs, acceleration and body-frame angular velocity only; other benchmark sensors excluded. Shoe angular-velocity physical units are undocumented in release metadata.'
    else:raise ValueError(dataset)
    length=int(x.shape[1]/rate*20)
    values=resample(values,length,axis=1).astype(np.float32)
    if length<200:values=np.pad(values,((0,0),(0,200-length),(0,0)),mode='wrap')
    return values[:,:200],joints,note


class FineTune(nn.Module):
    def __init__(self,backbone,joints,channels,classes):
        super().__init__();self.backbone=backbone;self.joints=joints;self.channels=channels
        self.head=nn.Linear(512,classes)
    def forward(self,x):
        b,t,_=x.shape
        full=x.new_zeros((b,t,22,6))
        full[:,:,self.joints]=x.reshape(b,t,len(self.joints),6)
        if self.channels==3:full=full[:,:,:,:3]
        elif self.channels==12:
            # Match the author's flattened concatenation and reshape exactly.
            flat=full.flatten(2)
            spectrum=torch.abs(torch.stft(flat.permute(0,2,1).reshape(-1,t),n_fft=25,hop_length=28,
                                          onesided=False,center=True,return_complex=True))
            spectrum=spectrum.reshape(b,132,-1).permute(0,2,1)
            full=torch.cat([flat,spectrum],dim=-1).reshape(b,t,22,12)
        inputs=full.permute(0,3,1,2).unsqueeze(-1)
        return self.head(self.backbone(inputs).flatten(1))


def prepare_model(x,train,validation,out,a,device):
    import importlib.util
    source=HERE/'source/unimts'
    spec=importlib.util.spec_from_file_location('official_unimts_stgcn',source/'model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    checkpoint=HERE/'reference/UniMTS.pth'
    expected='a9858c0084d936655240407e30ff9db9adeded6a67dc5650e3f667578e93b220'
    if sha(checkpoint)!=expected:raise ValueError('UniMTS author checkpoint checksum mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'state_dict' in state:state=state['state_dict']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    imu={k.removeprefix('acc.'):v for k,v in state.items() if k.startswith('acc.')}
    if not imu:raise ValueError('Author checkpoint has no acc branch; requires an audited key mapping')
    channels=len(imu['data_bn.weight'])//22
    if channels not in [3,6,12]:raise ValueError(f'Unknown UniMTS channel contract: {channels}')
    backbone=m.ST_GCN_18(in_channels=channels)
    backbone.load_state_dict(imu,strict=True)
    values,joints,note=adapt(x,a.dataset)
    classes=int(np.load(HERE/'data'/a.dataset/'y.npy').max()+1)
    model=FineTune(backbone,joints,channels,classes)
    return model,{'optimizer':'Adam','lr':.0001,'batch_size':32,'weight_decay':0.,'loss':'cross_entropy',
        'architecture':'official UniMTS ST_GCN_18 + author full-shot 512-dimensional linear classification head',
        'official_repo':'https://github.com/xiyuanzh/UniMTS',
        'official_commit':json.loads((source/'SOURCE.json').read_text())['commit'],
        'implementation_type':'official_code_sensor_adapter','model_sha256':sha(source/'model.py'),
        'checkpoint_sha256':expected,'checkpoint_url':'https://huggingface.co/xiyuanz/UniMTS/resolve/main/checkpoint/UniMTS.pth',
        'checkpoint_input_channels':channels,
        'checkpoint_modalities':{3:'accelerometer only',6:'accelerometer and gyroscope',12:'accelerometer/gyroscope with author STFT concatenation'}[channels],
        'input':'B,T,C','arithmetic':'float32','sensor_adapter':note,'joint_list':joints,
        'resampling':'author scipy Fourier resampling to 20 Hz; author wrap padding/truncation to 200 steps',
        'normalization_note':'Standard training-only z-score before the author physical sensor adapter; differs from paper calibration physical-units inputs',
        'calibration':'sensor-subset/native-window extension; fresh paper-protocol calibration pending',
        '_input_array':values}
