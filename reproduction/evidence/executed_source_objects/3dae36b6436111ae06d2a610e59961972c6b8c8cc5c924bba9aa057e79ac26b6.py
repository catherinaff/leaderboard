"""Author OneLLM IMU branch, frozen zero-shot classification extension.

This is deliberately separate from the supervised training queue. A shared
checkpoint is loaded once; no trained model or five independent fits are
claimed. Dataset class names are provided in the prompt, never sample labels.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import resample_poly

HERE=Path(__file__).resolve().parent


def flash_sdpa(q,k,v,dropout_p=0.,causal=False,softmax_scale=None,**kwargs):
    """FlashAttention bottom-right causal alignment, including cached tokens."""
    q,k,v=(a.transpose(1,2) for a in (q,k,v))
    mask=None
    if causal:
        nq,nk=q.shape[-2],k.shape[-2]
        mask=torch.arange(nk,device=q.device)[None,:]<=torch.arange(nq,device=q.device)[:,None]+nk-nq
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=dropout_p,scale=softmax_scale)
    return out.transpose(1,2)


def imu_window(data,dataset):
    indices=[6,7,8,3,4,5] if dataset=='uci_har' else list(range(6))
    rate={'uci_har':50,'pamap2':100,'usc_had':100,'realworld_har':50}[dataset]
    pair=resample_poly(np.asarray(data[:,indices]),200,rate,axis=0)
    # Author Ego4D IMU utility: resample to 200 Hz and zero-pad/truncate 400.
    value=np.zeros((400,6),dtype='float32');n=min(len(pair),400);value[:n]=pair[:n]
    return value.T


class VisualOnly(nn.Module):
    def __init__(self,visual):super().__init__();self.visual=visual


def load_author(device='cuda:1'):
    from unittest.mock import patch
    import open_clip
    import torch.distributed as dist
    import fairscale.nn.model_parallel.initialize as fs_init
    root=HERE/'vendor/onellm';assets=HERE/'assets_20261113/onellm'
    if not (assets/'ASSETS.json').exists():raise FileNotFoundError('Verified official OneLLM assets are still downloading')
    if not dist.is_initialized():
        handle=tempfile.NamedTemporaryFile(prefix='onellm-process-group-',delete=False);handle.close()
        dist.init_process_group('gloo',init_method='file://'+handle.name,rank=0,world_size=1)
    if not fs_init.model_parallel_is_initialized():fs_init.initialize_model_parallel(1)
    flash=ModuleType('flash_attn');flash.flash_attn_func=flash_sdpa;sys.modules['flash_attn']=flash
    # Load the unchanged component/tokenizer modules in a private namespace.
    for name,path in [('one_author',root/'model'),('one_author.LLM',root/'model/LLM')]:
        package=ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
    filename=root/'model/LLM/onellm.py';code=filename.read_text()
    before="self.modals = ['image', 'video', 'audio', 'point', 'rgbd', 'rgbn', 'fmri', 'imu']"
    assert code.count(before)==1;code=code.replace(before,"self.modals = ['imu']")
    module=ModuleType('one_author.LLM.onellm');module.__file__=str(filename);sys.modules[module.__name__]=module
    exec(compile(code,str(filename),'exec'),module.__dict__)
    from one_author.tokenizer import Tokenizer
    tokenizer=Tokenizer(str(assets/'tokenizer.model'))
    params=json.loads((assets/'params.json').read_text())
    params.update(vocab_size=tokenizer.n_words,max_seq_len=512,max_batch_size=1)
    original=open_clip.create_model_and_transforms
    def architecture(*args,**kwargs):
        kwargs['pretrained']=None
        clip,train,test=original(*args,**kwargs)
        return VisualOnly(clip.visual),train,test
    dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with patch.object(open_clip,'create_model_and_transforms',architecture):model=module.Transformer(module.ModelArgs(**params))
    finally:torch.set_default_dtype(dtype)
    checkpoint=assets/'consolidated.00-of-01.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if 'model' in state:state=state['model']
    # Original consolidated weights may carry the MetaModel `llma.` prefix.
    state={key.removeprefix('llma.'):value for key,value in state.items()}
    own=model.state_dict();selected={key:value for key,value in state.items() if key in own}
    unused={'clip.visual.conv1.weight','clip.visual.positional_embedding','clip.visual.proj'}
    missing=set(own)-set(selected)-unused
    if missing:raise ValueError(f'OneLLM checkpoint lacks used IMU/shared tensors: {sorted(missing)[:20]}')
    for key,value in selected.items():
        if own[key].shape!=value.shape:raise ValueError(f'OneLLM shape mismatch: {key}')
    model.load_state_dict(selected,strict=False);del state,selected,own
    model.requires_grad_(False);model.eval();model.to(device)
    provenance=dict(source=json.loads((root/'SOURCE.json').read_text()),assets=json.loads((assets/'ASSETS.json').read_text()),
        compatibility='Keep only IMU modality modules; retain all used shared CLIP/resampler/router/LLaMA weights. PyTorch SDPA with original causal alignment; architecture-only CLIP init, then strict used-weight coverage.',
        loaded_tensors=len(model.state_dict()),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        variant_label='Frozen author checkpoint · zero-shot HAR adapter',
        classification='A–L class-letter next-token logits; original generative IMU model, no supervised classifier training.')
    return model,tokenizer,provenance


def classify(model,tokenizer,windows,classes,device):
    letters=[chr(65+i) for i in range(len(classes))]
    descriptions='\n'.join(f'{letter}: {label}' for letter,label in zip(letters,classes))
    prompt=('You are given wearable accelerometer and gyroscope measurements. Identify the activity.\n'
            +descriptions+'\nAnswer with the single corresponding capital letter.\nAnswer:')
    tokens=tokenizer.encode(prompt,bos=True,eos=False)
    choices=[]
    for letter in letters:
        # LLaMA SentencePiece treats leading word-space consistently here.
        ids=tokenizer.encode(letter,bos=False,eos=False)
        if len(ids)!=1:raise ValueError(f'Class letter must be a single token: {letter} {ids}')
        choices.append(ids[0])
    assert len(tokens)+32<512
    output=[]
    with torch.inference_mode():
        for window in windows:
            x=torch.tensor(window[None],device=device,dtype=torch.float16)
            ids=torch.tensor([tokens],device=device,dtype=torch.long)
            logits=model.forward_inference(ids,0,image=x,modal='imu')[:,choices]
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite OneLLM logits')
            output.append(logits.cpu().numpy()[0])
    return np.stack(output)


def main(args):
    from api_adapters import prepare_inputs,dataset_metadata,atomic_json
    from sklearn.metrics import accuracy_score,f1_score
    torch.cuda.set_device(args.gpu);device=f'cuda:{args.gpu}'
    torch.cuda.set_per_process_memory_fraction(.85,args.gpu)
    model,tokenizer,provenance=load_author(device)
    datasets=[args.dataset] if args.dataset else ['uci_har','usc_had','pamap2','realworld_har']
    for dataset in datasets:
        metadata=dataset_metadata(dataset);directory=HERE/'data'/dataset
        x=np.load(directory/'x.npy',mmap_mode='r');y=np.load(directory/'y.npy')
        folds=sorted((HERE/'manifests/splits'/dataset/'seed_42').glob('*.npz'))
        if args.smoke:folds=folds[:1]
        for fold in folds:
            out=HERE/('smoke_zero_shot' if args.smoke else 'zero_shot_runs')/'onellm'/dataset/'seed_42'/fold.stem
            if (out/'result.json').exists():continue
            start=time.time();split=np.load(fold);tr=split['train'];te=split['test']
            raw,_,scaler=prepare_inputs(x,tr)
            indices=te[:2] if args.smoke else te
            windows=[imu_window(raw[int(i)],dataset) for i in indices]
            logits=classify(model,tokenizer,windows,metadata['classes'],device)
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'predictions.npz',test_indices=indices,y_true=y[indices],logits=logits,y_pred=logits.argmax(1))
            atomic_json(out/'result.json',dict(status='smoke_complete' if args.smoke else 'complete_zero_shot',
                method='onellm',dataset=dataset,seed=42,fold=fold.stem,n=len(indices),recipe=provenance,
                counts=dict(train_for_imputation=len(tr),test=len(te)),scaler=scaler,
                test=dict(accuracy=float(accuracy_score(y[indices],logits.argmax(1))),f1_macro=float(f1_score(y[indices],logits.argmax(1),labels=list(range(len(metadata['classes']))),average='macro',zero_division=0))),
                runtime_seconds=time.time()-start,protocol='Frozen zero-shot evaluation; seed42 split only, no five-fit claim. Separate from supervised canonical candidate scores.'))
            print('COMPLETE',dataset,fold.stem,len(indices),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--dataset')
    main(parser.parse_args())
