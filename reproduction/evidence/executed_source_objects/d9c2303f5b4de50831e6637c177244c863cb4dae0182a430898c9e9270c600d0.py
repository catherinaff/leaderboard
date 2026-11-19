"""Author foundation-model adapters on the frozen HAR windows and splits.

Original GPT4TS, CALF and UniTS classification branches are retained. TimeLLM
uses its author's GPT-2 forecast branch followed by an explicit HAR readout.
Frozen pretrained weights are referenced once instead of duplicated per fold.
"""
from contextlib import ExitStack,contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from prepare import HERE,sha

ASSETS=HERE/'assets_20261113'
GPT2=ASSETS/'gpt2'


def source(name):
    directory=HERE/'vendor'/name
    return directory,json.loads((directory/'SOURCE.json').read_text())


def load_file(file,name,changes=None):
    code=file.read_text()
    for before,after in (changes or {}).items():
        if code.count(before)!=1:raise ValueError(f'Compatibility patch does not match {file}: {before}')
        code=code.replace(before,after)
    module=type(sys)(name);module.__file__=str(file);sys.modules[name]=module
    exec(compile(code,str(file),'exec'),module.__dict__)
    return module,hashlib.sha256(code.encode()).hexdigest()


@contextmanager
def local_gpt2(*classes):
    """Redirect original hard-coded HF names to the verified local snapshot."""
    if not (GPT2/'ASSETS.json').exists():raise FileNotFoundError('Verified GPT2 assets are not ready')
    with ExitStack() as stack:
        for cls in classes:
            original=cls.from_pretrained.__func__
            def loader(actual_cls,*args,_original=original,**kwargs):
                kwargs['local_files_only']=True
                return _original(actual_cls,str(GPT2),*args[1:],**kwargs)
            stack.enter_context(patch.object(cls,'from_pretrained',classmethod(loader)))
        yield


def asset_metadata():
    value=json.loads((GPT2/'ASSETS.json').read_text())
    return dict(repo=value['repo'],revision=value['revision'],weights_sha256=value['files']['model.safetensors']['sha256'])


class CompactAdapter(nn.Module):
    """Exact trainable state; immutable base weights are rebuilt from SHA pins."""
    def checkpoint_state_dict(self):
        trainable={name for name,p in self.named_parameters(remove_duplicate=False) if p.requires_grad}
        mutable={name for name,_ in self.named_buffers() if any(t in name for t in ['running_mean','running_var','num_batches_tracked'])}
        return {name:value for name,value in self.state_dict().items() if name in trainable or name in mutable}

    def load_checkpoint_state_dict(self,state):
        expected={name for name,p in self.named_parameters(remove_duplicate=False) if p.requires_grad}
        missing=expected-set(state)
        if missing:raise ValueError(f'Missing trainable parameters: {sorted(missing)[:8]}')
        loaded=self.load_state_dict(state,strict=False)
        if loaded.unexpected_keys:raise ValueError(loaded.unexpected_keys)


class GPT4TSAdapter(CompactAdapter):
    def __init__(self,model):super().__init__();self.core=model
    def forward(self,x):return self.core(x,torch.ones(x.shape[:2],dtype=torch.bool,device=x.device))


class CALFAdapter(CompactAdapter):
    def __init__(self,model,loss):super().__init__();self.core=model;self.loss=loss
    def forward(self,x):return self.core(x)['outputs_time']
    def training_loss(self,x,y):return self.loss(self.core(x),y)


class UniTSAdapter(CompactAdapter):
    def __init__(self,model):super().__init__();self.core=model
    def forward(self,x):return self.core(x,torch.ones(x.shape[:2],device=x.device),task_id=0,task_name='classification')


class IMU2CLIPAdapter(CompactAdapter):
    def __init__(self,model):super().__init__();self.core=model
    def forward(self,x):return self.core(x.transpose(1,2))


class ConstantMapping(nn.Module):
    def __init__(self,embedding):super().__init__();self.register_buffer('embedding',embedding)
    def forward(self,ignored):return self.embedding.T


class TimeLLMAdapter(CompactAdapter):
    def __init__(self,model,channels,classes):
        super().__init__();self.core=model;self.head=nn.Linear(model.pred_len*channels,classes)
    def forward(self,x):
        # Matches the original caller's bf16 autocast, including its explicit
        # bfloat16 patch input. The added readout is trained with HAR labels.
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16):
            forecast=self.core(x,None,None,None)
            result=self.head(forecast.flatten(1))
        return result.float()
    def checkpoint_state_dict(self):
        state=super().checkpoint_state_dict()
        state={name:value for name,value in state.items() if not name.startswith('core.mapping_layer.')}
        with torch.no_grad(),torch.autocast(device_type=self.core.word_embeddings.device.type,dtype=torch.bfloat16):
            embedding=self.core.mapping_layer(self.core.word_embeddings.T).T.detach()
        # The frozen word embeddings are the sole input to this 50M-parameter
        # map. Saving its exact output is lossless graph constant folding for
        # inference, reducing a checkpoint by ~200 MB. Training is unchanged.
        state['__source_embedding_cache__']=embedding
        return state
    def load_checkpoint_state_dict(self,state):
        state=dict(state);embedding=state.pop('__source_embedding_cache__')
        self.core.mapping_layer=ConstantMapping(embedding)
        super().load_checkpoint_state_dict(state)


def build_gpt4ts(x,classes,device):
    directory,info=source('gpt4ts');root=directory/'Classification/src';sys.path.insert(0,str(root))
    file=root/'models/gpt4ts.py'
    module,patched_sha=load_file(file,'author_gpt4ts',{
        "device = torch.device('cuda:{}'.format(0))":f"device = torch.device('{device}')"})
    config=dict(patch_size=8,stride=4,d_model=768,dropout=.1)
    data=SimpleNamespace(max_seq_len=x.shape[1],feature_df=SimpleNamespace(shape=(0,x.shape[2])),class_names=list(range(classes)))
    with local_gpt2(module.GPT2Model):model=module.gpt4ts(config,data)
    return GPT4TSAdapter(model),dict(optimizer='RAdam',lr=.001,batch_size=32,weight_decay=0.,loss='cross_entropy',
        architecture='author classification gpt4ts: patch8/stride4, GPT2 first six blocks, frozen attention/MLP, trainable LN/positions/readout',
        model_sha256=sha(file),compatibility_sha256=patched_sha,compatibility='Replace only hard-coded cuda:0 with selected worker GPU; redirect HF asset name locally.',**provenance(info))


def build_calf(x,classes,device):
    directory,info=source('calf');sys.path.insert(0,str(directory))
    import peft
    from models.GPT2_arch import AccustumGPT2Model
    module,patched_sha=load_file(directory/'models/CALF.py','author_calf')
    # The released model references this helper but omits it from its import.
    module.get_peft_model=peft.get_peft_model
    basis=ASSETS/'gpt2/wte_pca_500.pt'
    if not basis.exists():raise FileNotFoundError('Author PCA word basis is not prepared')
    config=SimpleNamespace(task_name='classification',pred_len=0,r=8,lora_alpha=32,lora_dropout=.1,
        seq_len=x.shape[1],enc_in=x.shape[2],num_class=classes,d_model=768,gpt_layers=6,word_embedding_path=str(basis))
    with local_gpt2(AccustumGPT2Model):model=module.Model(config,device)
    from utils.cmLoss import cmLoss
    loss=cmLoss('l1','l1','ce','classification',feature_w=.01,output_w=1.,task_w=1.)
    return CALFAdapter(model,loss),dict(optimizer='Adam',lr=.0001,batch_size=16,weight_decay=0.,
        loss='author cmLoss: CE(time labels) + 1*L1(time/text outputs) + .01*layer-weighted L1(feature alignment)',
        architecture='author CALF classification: two GPT2 six-block branches, rank8 LoRA, PCA500 cross-modal encoder and author alignment losses',
        model_sha256=sha(directory/'models/CALF.py'),compatibility_sha256=patched_sha,pca_sha256=sha(basis),
        compatibility='Supply omitted get_peft_model import; tensor-only serialization for the unchanged PCA numeric basis; local verified GPT2 checkpoint.',**provenance(info))


def build_units(x,classes,device):
    directory,info=source('units');file=directory/'models/UniTS.py';module,_=load_file(file,'author_units')
    config=SimpleNamespace(prompt_num=10,patch_len=16,stride=16,e_layers=3,d_model=64,n_heads=8,dropout=.1)
    tasks=[['har',dict(dataset='har',task_name='classification',enc_in=x.shape[2],seq_len=x.shape[1],num_class=classes)]]
    model=module.Model(config,tasks)
    checkpoint=ASSETS/'units/units_x64_supervised_checkpoint.pth'
    if not checkpoint.exists():raise FileNotFoundError('Original UniTS supervised pretrained checkpoint is missing')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key in ['model','state_dict']:
        if isinstance(state,dict) and key in state and isinstance(state[key],dict):state=state[key];break
    state={key.removeprefix('module.'):value for key,value in state.items()}
    task_prefixes=('prompt_tokens.','mask_tokens.','cls_tokens.','category_tokens.')
    own=model.state_dict();selected={key:value for key,value in state.items() if key in own and not key.startswith(task_prefixes)}
    for key,value in selected.items():
        if value.shape!=own[key].shape:raise ValueError(f'UniTS pretrained shape mismatch: {key}')
    loaded=model.load_state_dict(selected,strict=False)
    missing=[key for key in loaded.missing_keys if not key.startswith(task_prefixes)]
    if missing:raise ValueError(f'UniTS missing shared weights: {missing[:10]}')
    return UniTSAdapter(model),dict(optimizer='AdamW',lr=.0001,batch_size=64,weight_decay=.00001,loss='cross_entropy',
        architecture='author UniTS x64 shared backbone, three blocks, original classification/category tokens; full fine-tuning',
        model_sha256=sha(file),checkpoint_sha256=sha(checkpoint),loaded_shared_tensors=len(selected),
        compatibility='Author new-dataset transfer: keep shared weights, initialize new dataset/task/category tokens; all native channels retained.',**provenance(info))


def build_timellm(x,classes,device):
    directory,info=source('timellm');sys.path.insert(0,str(directory));file=directory/'models/TimeLLM.py'
    module,_=load_file(file,'author_timellm')
    config=SimpleNamespace(task_name='long_term_forecast',pred_len=16,seq_len=x.shape[1],enc_in=x.shape[2],
        d_ff=128,llm_dim=768,patch_len=16,stride=8,llm_model='GPT2',llm_layers=6,
        prompt_domain=True,content='Wearable inertial measurements of human daily activities from accelerometers and gyroscopes.',
        dropout=.1,d_model=32,n_heads=8)
    with local_gpt2(module.GPT2Config,module.GPT2Model,module.GPT2Tokenizer):model=module.Model(config)
    return TimeLLMAdapter(model,x.shape[2],classes),dict(optimizer='Adam',lr=.0001,batch_size=2 if x.shape[2]>27 else 4,weight_decay=0.,loss='cross_entropy',
        architecture='author-supported TimeLLM GPT2 six-block forecast branch, word reprogramming, original prompt/statistics/denormalization; linear HAR readout over 16 predicted steps',
        model_sha256=sha(file),variant_label='GPT-2 · classification adapter',
        compatibility='Task extension, not the original forecasting task or a LLaMA paper reproduction. Original forecast arithmetic retained; added linear class readout.',
        checkpoint_note='Inference checkpoint constant-folds the frozen word-input mapping to its exact bf16 output; optimizer-state resumption is not provided.',**provenance(info))


def physical_pair(x,dataset):
    """One native physical acc/gyro pair; UCI total acceleration includes gravity."""
    if dataset=='uci_har':return np.concatenate([x[:,:,6:9],x[:,:,3:6]],axis=2),50,'UCI total_acc + body_gyro'
    rate=dict(pamap2=100,usc_had=100,realworld_har=50)[dataset]
    position=dict(pamap2='hand',usc_had='hip',realworld_har='chest')[dataset]
    return x[:,:,:6],rate,position+' accelerometer + gyroscope'


def build_imu2clip(x,classes,device,dataset):
    from scipy.signal import resample_poly
    directory,info=source('imu2clip')
    # Lightning is used only as an nn.Module base in these released components;
    # the original Lightning trainer is unnecessary for the frozen split runner.
    change={'import pytorch_lightning as pl':'from types import SimpleNamespace\npl = SimpleNamespace(LightningModule=nn.Module)'}
    module,compatibility_sha=load_file(directory/'lib/imu_models.py','author_imu2clip_encoder',change)
    heads,_=load_file(directory/'lib/classification_head.py','author_imu2clip_heads',change)
    encoder=module.MW2StackRNNPooling(size_embeddings=512)
    assets=ASSETS/'imu2clip';manifest=json.loads((assets/'ASSETS.json').read_text())
    name='shane_models/i2c_s_i_t_t_ie_mw2_w_2.5_master_imu_encoder.pt'
    file=assets/name
    if sha(file)!=manifest['files'][name]['sha256']:raise ValueError('IMU2CLIP checkpoint checksum changed')
    encoder.load_state_dict(torch.load(file,map_location='cpu',weights_only=True),strict=True)
    model=heads.Head(encoder,512,classes)
    values,rate,sensor=physical_pair(x,dataset)
    values=resample_poly(values,200,rate,axis=1).astype(np.float32)
    if values.shape[1]<1000:values=np.pad(values,((0,0),(0,1000-values.shape[1]),(0,0)),mode='constant')
    values=values[:,:1000]
    # The released MW2 model's GRU input is 32; its three author conv/pool
    # stages produce 32 positions from the original 5 s x 200 Hz input.
    assert values.shape[1:]==(1000,6)
    return IMU2CLIPAdapter(model),dict(optimizer='Adam',lr=.0001,batch_size=32,weight_decay=0.,loss='cross_entropy',
        architecture='author MW2StackRNNPooling(512) + author linear Head; strict-loaded pretrained encoder, both encoder and head fine-tuned',
        model_sha256=sha(directory/'lib/imu_models.py'),compatibility_sha256=compatibility_sha,
        checkpoint_sha256=manifest['files'][name]['sha256'],checkpoint_repo=manifest['repo'],checkpoint_revision=manifest['revision'],
        sensor_adapter=sensor+'; training-only z-score, 200 Hz polyphase resampling, zero pad/truncate to author 1000 steps. Other physical devices/magnetometers are excluded.',
        compatibility='Retain original encoder/head arithmetic; replace only the Lightning shell base with nn.Module.',
        provenance_note='Public community distribution of the original IMU2CLIP encoder; all original parameter names and shapes match strictly. Training-corpus provenance still needs paper calibration.',
        _input_array=values,**provenance(info))


def provenance(info):
    return dict(official_repo=info['repo'],official_commit=info['commit'],source_archive_sha256=info['archive_sha256'],
        implementation_type='official_code_classification_adapter',adapter_sha256=sha(__file__))


def prepare_model(x,train,validation,out,a,device):
    classes=int(np.load(HERE/'data'/a.dataset/'y.npy').max()+1)
    builders=dict(gpt4ts=build_gpt4ts,calf=build_calf,units=build_units,timellm=build_timellm)
    if a.method=='imu2clip':model,recipe=build_imu2clip(x,classes,device,a.dataset)
    else:model,recipe=builders[a.method](x,classes,device)
    recipe.setdefault('sensor_adapter','All canonical native channels and window samples; training-only imputation and scaling.')
    recipe.update(input='B,T,C',arithmetic='bf16 autocast in TimeLLM' if a.method=='timellm' else 'float32',
        checkpoint_policy='Trainable parameters plus mutable normalization state; immutable pretrained assets referenced by checksum.',
        frozen_assets=asset_metadata() if a.method in ['gpt4ts','calf','timellm'] else {},
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        calibration='Unified canonical-data candidate; exact paper-protocol calibration pending.')
    return model,recipe
