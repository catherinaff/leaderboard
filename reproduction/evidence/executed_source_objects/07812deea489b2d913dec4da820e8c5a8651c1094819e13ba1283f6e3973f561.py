"""Paper reimplementation of IJCAI 2025 RobustHAR, NOT author code.

See ROBUSTHAR_AUDIT.md for equations, unspecified choices and calibration limits.
All native features are retained, including mixed sensor types in OPPORTUNITY.
"""
import copy
import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare import HERE, atomic_json, sha


def sensor_groups(dataset,channels):
    if dataset=='pamap2':
        # Anatomical order: chest, wrist, ankle; original data order hand/chest/ankle.
        return [list(range(9,18)),list(range(0,9)),list(range(18,27))]
    if dataset=='opportunity_a':
        deleted={i for a,b in [(46,50),(59,63),(72,76),(85,89),(98,102),(134,243),(244,249)] for i in range(a,b)}
        selected=[i for i in range(250) if i not in deleted][1:114]
        names={}
        for line in (HERE/'reference/opportunity_column_names.txt').read_text().splitlines():
            m=re.match(r'Column: (\d+) (Accelerometer|InertialMeasurementUnit) (\S+)',line)
            if m:names[int(m[1])-1]=(m[2],m[3])
        by_sensor={}
        for i,raw in enumerate(selected):by_sensor.setdefault(names[raw],[]).append(i)
        # Height, then left/right/name tie-breaks. This A is a fixed permutation;
        # the paper does not supply its numerical position matrix.
        heights={'BACK':0,'LUA^':1,'LUA_':1,'RUA^':1,'RUA_':1,'LUA':1,'RUA':1,
                 'LLA':2,'RLA':2,'LWR':2,'RWR':2,'LH':3,'RH':3,'HIP':4,
                 'RKN^':5,'RKN_':5,'L-SHOE':6,'R-SHOE':6}
        groups=[by_sensor[k] for k in sorted(by_sensor,key=lambda k:(heights[k[1]],k))]
        assert sorted(i for g in groups for i in g)==list(range(channels))
        return groups
    return [list(range(channels))]


class Branch(nn.Module):
    def __init__(self,features,length,hidden=128):
        super().__init__()
        self.input=nn.Linear(features,hidden)
        pe=torch.zeros(length,hidden)
        positions=torch.arange(length).float()[:,None]
        div=torch.exp(torch.arange(0,hidden,2).float()*(-math.log(10000.)/hidden))
        pe[:,0::2]=torch.sin(positions*div);pe[:,1::2]=torch.cos(positions*div)
        self.register_buffer('position',pe)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,8,512,.1,batch_first=True),1)
        self.decoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,8,512,.1,batch_first=True),1)
        self.output=nn.Linear(hidden,features)

    def encode(self,x):return self.encoder(self.input(x)+self.position)
    def forward(self,x):
        h=self.encode(x)
        return self.output(self.decoder(h))


class RobustHAR(nn.Module):
    def __init__(self,dataset,length,channels,classes):
        super().__init__()
        self.groups=sensor_groups(dataset,channels)
        width=max(map(len,self.groups))
        mapping=torch.full((len(self.groups),width),channels,dtype=torch.long)
        valid=torch.zeros_like(mapping,dtype=torch.bool)
        for i,g in enumerate(self.groups):mapping[i,:len(g)]=torch.tensor(g);valid[i,:len(g)]=True
        self.register_buffer('mapping',mapping);self.register_buffer('valid',valid)
        self.branches=nn.ModuleList([Branch(mapping.numel(),length) for _ in range(3)])
        self.fusion=nn.ModuleList([nn.MultiheadAttention(128,8,.1,batch_first=True) for _ in range(3)])
        self.classifier=nn.Linear(128,classes)

    def layout(self,x):
        # Pad nonexistent measurement channels, never discard real channels.
        x=F.pad(x,(0,1))
        return x[:,:,self.mapping]

    def masks(self,x,rate=.3,generator=None):
        b,t,n,c=x.shape
        # Three views: segment-wise contiguous block, channel-wise, sensor-wise.
        seg=torch.ones((b,t,1,1),device=x.device)
        length=max(1,round(t*rate))
        starts=torch.randint(t-length+1,(b,),device=x.device,generator=generator)
        positions=torch.arange(t,device=x.device)[None,:]
        seg[:,:,0,0]=~((positions>=starts[:,None])&(positions<starts[:,None]+length))
        channel=(torch.rand((b,1,1,c),device=x.device,generator=generator)>=rate).float()
        sensor=(torch.rand((b,1,n,1),device=x.device,generator=generator)>=rate).float()
        return [seg.expand_as(x),channel.expand_as(x),sensor.expand_as(x)]

    def pretrain_loss(self,raw,generator=None):
        x=self.layout(raw)
        masks=self.masks(x,generator=generator)
        completed=[];rec=raw.new_zeros(())
        valid=self.valid.expand_as(x)
        for branch,mask in zip(self.branches,masks):
            pred=branch((x*mask).flatten(2)).reshape_as(x)
            # Eq.5 is reconstruction at masked portions, excluding layout padding.
            selected=(mask==0)&valid
            if selected.any():rec=rec+F.l1_loss(pred[selected],x[selected])
            completed.append(torch.where(mask.bool(),x,pred)*valid)
        # Eq.6 says minimize "sim" yet mentions cosine similarity. Interpret it
        # as cosine distance (1-cos), otherwise minimization enforces disagreement.
        con=sum((1-F.cosine_similarity(completed[i].flatten(1),completed[j].flatten(1),dim=1)).mean()
                for i,j in [(0,1),(0,2),(1,2)])
        return .5*rec+.5*con

    def freeze_encoders(self):
        for branch in self.branches:
            for p in branch.parameters():p.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        for branch in self.branches:
            if not any(p.requires_grad for p in branch.parameters()):branch.eval()
        return self

    def forward(self,raw):
        x=self.layout(raw).flatten(2)
        features=[branch.encode(x) for branch in self.branches]
        fused=[]
        for layer,(i,j) in zip(self.fusion,[(0,1),(0,2),(1,2)]):
            a,b=features[i],features[j]
            # Literal Eq.9 cross-normalized self-Gram, then correlation-guided
            # cross-attention. Eqs.9-11 do not specify query/key/value projections.
            an=a/b.norm(dim=1,keepdim=True).clamp_min(1e-6)
            bn=b/a.norm(dim=1,keepdim=True).clamp_min(1e-6)
            ra=an.transpose(1,2)@an;rb=bn.transpose(1,2)@bn
            aa=layer(a@ra/math.sqrt(128),b,b,need_weights=False)[0]
            ab=layer(b@rb/math.sqrt(128),a,a,need_weights=False)[0]
            fused.append((aa*ab).mean(1))
        return self.classifier(torch.stack(fused).mean(0))


def prepare_model(x,train,validation,out,a,device):
    model=RobustHAR(a.dataset,x.shape[1],x.shape[2],int(np.load(HERE/'data'/a.dataset/'y.npy').max()+1)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    gx=torch.as_tensor(x,device=device)
    best=float('inf');stale=0;history=[]
    for epoch in range(1,201):
        model.train();losses=[]
        for ids in np.array_split(np.random.permutation(train),math.ceil(len(train)/64)):
            opt.zero_grad(set_to_none=True);loss=model.pretrain_loss(gx[ids])
            if not torch.isfinite(loss):raise ValueError('non-finite pretraining loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        model.eval();vals=[]
        rng=torch.Generator(device=device).manual_seed(a.seed+73000)
        with torch.no_grad():
            for ids in np.array_split(validation,math.ceil(len(validation)/64)):
                vals.append(float(model.pretrain_loss(gx[ids],generator=rng)))
        val=float(np.mean(vals));history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':val})
        atomic_json(out/'pretrain_history.json',history)
        if val<best-.0005:
            best=val;stale=0;torch.save(model.state_dict(),out/'pretrain_best.pt')
        else:stale+=1
        if epoch==1 or epoch%5==0:print('RobustHAR pretrain',a.dataset,a.seed,a.fold,epoch,history[-1],flush=True)
        if stale>=25:break
    model.load_state_dict(torch.load(out/'pretrain_best.pt',weights_only=True,map_location=device))
    model.freeze_encoders()
    return model,{'optimizer':'Adam','lr':.001,'batch_size':32,'weight_decay':0.,'loss':'cross_entropy',
        'architecture':'paper reimplementation: three 1-layer 8-head encoders/decoders; adaptive cross-attention fusion',
        'official_repo':None,'official_commit':None,'implementation_type':'paper_reimplementation',
        'paper':'https://www.ijcai.org/proceedings/2025/952','model_sha256':sha(__file__),
        'input':'B,T,C','arithmetic':'float32','max_epochs':100,
        'pretrain_epochs_run':epoch,'pretrain_best_validation_loss':best,'pretrain_sha256':sha(out/'pretrain_best.pt'),
        'sensor_groups':model.groups,'calibration':'not yet calibrated; implementation choices enumerated in ROBUSTHAR_AUDIT.md',
        'audit_sha256':sha(HERE/'ROBUSTHAR_AUDIT.md')}
