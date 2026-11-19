"""PyTorch port of author TF1 DeepSense operations, with native-window adapter.

Preserves frequency-major real/imag FFT, per-modality convolutions, TF SAME
fusion padding, reset-before GRU semantics and mean-over-valid-time pooling.
This port still requires paper-protocol calibration; it is not author code.
"""
from typing import List
import math
import torch
from torch import nn
from torch.nn import functional as F


@torch.jit.script
def gru_scan(xg:torch.Tensor,xc:torch.Tensor,wg:torch.Tensor,wc:torch.Tensor):
    h=xg.new_zeros((xg.size(0),wg.size(0)))
    result=torch.jit.annotate(List[torch.Tensor],[])
    for i in range(xg.size(1)):
        r,u=torch.sigmoid(xg[:,i]+h@wg).chunk(2,1)
        candidate=torch.tanh(xc[:,i]+(r*h)@wc)
        h=u*h+(1-u)*candidate
        result.append(h)
    return torch.stack(result,1)


class TFGRU(nn.Module):
    def __init__(self,inputs,hidden=120):
        super().__init__()
        self.gate=nn.Linear(inputs,2*hidden)
        self.candidate=nn.Linear(inputs,hidden)
        self.wg=nn.Parameter(torch.empty(hidden,2*hidden));self.wc=nn.Parameter(torch.empty(hidden,hidden))
        for w in [self.gate.weight,self.candidate.weight,self.wg,self.wc]:nn.init.xavier_uniform_(w)
        nn.init.ones_(self.gate.bias);nn.init.zeros_(self.candidate.bias)
    def forward(self,x):return gru_scan(self.gate(x),self.candidate(x),self.wg,self.wc)


def modality_groups(dataset,channels):
    if dataset!='opportunity_a':return [list(range(i,min(i+3,channels))) for i in range(0,channels,3)]
    # 12 accelerometers + 5 (acc/gyro/mag) body IMUs + 2 shoes
    # (Euler/nav-acc/body-acc/body-angvel/nav-angvel/compass).
    groups=[list(range(i,i+3)) for i in range(0,81,3)]
    for start in [81,97]:
        groups.extend([list(range(start+i,start+i+3)) for i in range(0,15,3)])
        groups.append([start+15])
    assert sorted(i for g in groups for i in g)==list(range(channels))
    return groups


class Modality(nn.Module):
    def __init__(self,dimensions):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1,64,(1,6*dimensions),stride=(1,2*dimensions)),
            nn.BatchNorm2d(64,eps=.001,momentum=.001),nn.ReLU(),nn.Dropout2d(.2),
            nn.Conv2d(64,64,(1,3)),nn.BatchNorm2d(64,eps=.001,momentum=.001),nn.ReLU(),nn.Dropout2d(.2),
            nn.Conv2d(64,64,(1,3)),nn.BatchNorm2d(64,eps=.001,momentum=.001),nn.ReLU())
    def forward(self,x):return self.layers(x.unsqueeze(1))


class DeepSense(nn.Module):
    def __init__(self,dataset,channels,classes):
        super().__init__()
        self.groups=modality_groups(dataset,channels)
        self.modalities=nn.ModuleList([Modality(len(g)) for g in self.groups])
        self.drop_input=nn.Dropout3d(.2)
        layers=[]
        for i,width in enumerate([8,6,4]):
            layers.extend([nn.Conv3d(64,64,(1,2,width),padding='same'),
                           nn.BatchNorm3d(64,eps=.001,momentum=.001),nn.ReLU()])
            if i<2:layers.append(nn.Dropout3d(.2))
        self.fusion=nn.Sequential(*layers)
        self.first=TFGRU(64*4*len(self.groups));self.second=TFGRU(120)
        self.dropout=nn.Dropout(.5);self.output=nn.Linear(120,classes)
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Conv3d)):
                nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output.weight);nn.init.zeros_(self.output.bias)

    def spectrum(self,x):
        length=x.shape[1];steps=math.ceil(length/10)
        x=F.pad(x,(0,0,0,steps*10-length)).reshape(x.shape[0],steps,10,x.shape[2])
        z=torch.fft.fft(x,dim=2)
        # Author HHAR preprocessing emits frequency -> channel -> real/imag.
        return torch.stack([z.real,z.imag],dim=-1)

    def forward(self,x):
        z=self.spectrum(x)
        features=[m(z[:,:,:,g,:].flatten(2)) for g,m in zip(self.groups,self.modalities)]
        x=torch.stack(features,dim=3)  # B,64,time,modality,4
        x=self.fusion(self.drop_input(x))
        x=x.permute(0,2,3,4,1).flatten(2)
        x=self.dropout(self.first(x));x=self.dropout(self.second(x))
        return self.output(x.mean(1))
