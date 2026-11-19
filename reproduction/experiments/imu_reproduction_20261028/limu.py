"""Author LIMU-BERT components, full native windows/channels, no test pretraining."""
import json
import math
import sys

import numpy as np
import torch
from torch import nn

from prepare import HERE, atomic_json, sha


class FrozenLimuGRU(nn.Module):
    def __init__(self,encoder,classifier):
        super().__init__();self.encoder=encoder;self.classifier=classifier
        for p in self.encoder.parameters():p.requires_grad_(False)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def chunk_logits(self,x):
        with torch.no_grad():h=self.encoder(x)
        b,t,d=h.shape;n=t//20
        full=h[:,:n*20].reshape(b*n,20,d)
        logits=self.classifier(full,self.training).reshape(b,n,-1)
        weights=h.new_full((n,),20.)
        if t%20:
            tail=self.classifier(h[:,n*20:],self.training).unsqueeze(1)
            logits=torch.cat([logits,tail],1)
            weights=torch.cat([weights,h.new_tensor([t%20])])
        return logits,weights/weights.sum()
    def training_loss(self,x,y):
        logits,weights=self.chunk_logits(x)
        labels=y[:,None].expand(-1,logits.shape[1]).reshape(-1)
        losses=nn.functional.cross_entropy(logits.flatten(0,1),labels,reduction='none').reshape(logits.shape[:2])
        return (losses*weights).sum(1).mean()
    def forward(self,x):
        logits,weights=self.chunk_logits(x)
        return (logits.softmax(-1)*weights[None,:,None]).sum(1).clamp_min(1e-12).log()


def prepare_model(x,train,validation,out,a,device):
    source=HERE/'source/LIMU-BERT-Public'
    sys.path.insert(0,str(source))
    from config import load_model_config,MaskConfig
    from models import LIMUBertModel4Pretrain,fetch_classifier
    from utils import Preprocess4Mask
    cfg=load_model_config('pretrain_base','base','v1',path_bert=str(source/'config/limu_bert.json'))
    cfg=cfg._replace(feature_num=x.shape[2],seq_len=x.shape[1])
    masker=Preprocess4Mask(MaskConfig.from_json(str(source/'config/mask.json')))
    encoder=LIMUBertModel4Pretrain(cfg).to(device)
    opt=torch.optim.Adam(encoder.parameters(),lr=.001)
    def masked(ids):
        rows=[masker(x[i]) for i in ids]
        return tuple(torch.as_tensor(np.stack([r[k] for r in rows]),device=device,
                     dtype=torch.long if k==1 else torch.float32) for k in range(3))
    # Freeze validation masking so early stopping compares identical examples.
    batch=a.pretrain_batch_size or 128
    fixed=[masked(ids) for ids in np.array_split(validation,math.ceil(len(validation)/batch))]
    best=float('inf');stale=0;history=[]
    for epoch in range(1,2 if getattr(a,'smoke',False) else 201):
        encoder.train();losses=[]
        for ids in np.array_split(np.random.permutation(train),math.ceil(len(train)/batch)):
            values,positions,targets=masked(ids)
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(encoder(values,positions),targets)
            if not torch.isfinite(loss):raise ValueError('non-finite LIMU pretraining loss')
            loss.backward();nn.utils.clip_grad_norm_(encoder.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        encoder.eval()
        with torch.no_grad():val=float(np.mean([float(nn.functional.mse_loss(encoder(v,p),t)) for v,p,t in fixed]))
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':val})
        atomic_json(out/'pretrain_history.json',history)
        if val<best-.0005:
            best=val;stale=0;torch.save(encoder.state_dict(),out/'pretrain_best.pt')
        else:stale+=1
        if epoch==1 or epoch%5==0:print('LIMU pretrain',a.dataset,a.seed,a.fold,epoch,history[-1],flush=True)
        if stale>=25:break
    encoder.load_state_dict(torch.load(out/'pretrain_best.pt',map_location=device,weights_only=True))
    encoder.output_embed=True
    clf_cfg=load_model_config('classifier_base_gru','gru','v2',path_classifier=str(source/'config/classifier.json'))
    classes=int(np.load(HERE/'data'/a.dataset/'y.npy').max()+1)
    classifier=fetch_classifier('gru',clf_cfg,input=cfg.hidden,output=classes).to(device)
    net=FrozenLimuGRU(encoder,classifier)
    info=json.loads((source/'SOURCE.json').read_text())
    return net,{'optimizer':'Adam','lr':.001,'batch_size':128,'weight_decay':0.,'loss':'cross_entropy',
        'architecture':'author LIMUBertModel4Pretrain + frozen encoder + author GRU v2 classifier',
        'official_repo':'https://github.com/dapowan/LIMU-BERT-Public','official_commit':info['commit'],
        'implementation_type':'official_code_native_input_adapter','model_sha256':sha(source/'models.py'),
        'input':'B,T,C','arithmetic':'float32','pretrain_epochs_run':epoch,'pretrain_batch_size':batch,'pretrain_best_validation_loss':best,
        'pretrain_sha256':sha(out/'pretrain_best.pt'),'encoder_config':cfg._asdict(),
        'adapter':'Native benchmark channels and sequence length, configured input/reconstruction layers; no channel truncation or resampling. Author GRU v2 receives 20-step embedding chunks; training uses chunk CE and window predictions average chunk probabilities. A short final chunk is weighted by its valid length.',
        'calibration':'new native-input adapter needs paper-protocol calibration; prior six-channel/low-label rows remain separate'}
