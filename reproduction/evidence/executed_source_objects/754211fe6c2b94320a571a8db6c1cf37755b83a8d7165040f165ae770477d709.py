"""Native SensorLLM two-stage adapter; shared frozen LLaMA/Chronos assets.

Stage one learns the original signal/trend-language alignment on independent
MHEALTH data. Stage two uses the author's classifier, Chronos tokenizer and
projector. The new RealWorld schema and input summaries are explicit adapters,
not claimed as an exact paper protocol. No LLM API is used for QA generation.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'vendor/sensorllm'
ASSETS=HERE/'assets_20261113'
sys.path.insert(0,str(SOURCE))


def original_qa_functions():
    notebook=json.loads((SOURCE/'mhealth_stage1.ipynb').read_text())
    scope=dict(np=np,pd=pd,random=random,re=re)
    for cell in [6,7]:
        source=''.join(notebook['cells'][cell]['source'])
        tree=ast.parse(source)
        # No notebook cells that read API keys, call an LLM, or write files.
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) or
            (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT_DICT' for t in n.targets))]
        exec(compile(ast.Module(body=selected,type_ignores=[]),f'{SOURCE}/mhealth_stage1.ipynb:cell{cell}','exec'),scope)
    return scope


def metadata(dataset):
    from api_adapters import dataset_metadata
    value=dataset_metadata(dataset)
    if dataset=='uci_har':
        value['channels']=[value['channels'][i] for i in [6,7,8,3,4,5]]
    if dataset=='pamap2':value['sampling_rate']=50
    return value


def input_windows(x,dataset):
    if dataset=='uci_har':return np.asarray(x[:,:,[6,7,8,3,4,5]])
    if dataset=='pamap2':return resample_poly(x,1,2,axis=1).astype('float32')
    return np.asarray(x,dtype='float32')


def config_entry(meta):
    entry=dict(channel_num=len(meta['channels']),sample_rate=meta['sampling_rate'],num_labels=len(meta.get('classes',[])),
               id2label={i:name for i,name in enumerate(meta.get('classes',[]))})
    # Insertion order fixes the exact physical-channel to special-token map.
    for name in meta['channels']:
        safe=re.sub('[^a-zA-Z0-9_]','_',name)
        entry[f'default_{safe}_start_token']=f'<{safe}_start>'
        entry[f'default_{safe}_end_token']=f'<{safe}_end>'
    return entry


def install_schema(model,tokenizer,meta,device):
    key='benchmark';model.get_model().ts_backbone_config[key]=config_entry(meta)
    model.initialize_tokenizer_ts_backbone_config(tokenizer,device,fix_llm=True,dataset=key)
    model.get_model().load_start_end_tokens(key)
    return model.get_model().ts_backbone_config[key]


def chronos_tokenizer():
    from transformers import AutoConfig
    from sensorllm.model.chronos_model import ChronosConfig
    config=AutoConfig.from_pretrained(str(ASSETS/'chronos'),local_files_only=True)
    value=ChronosConfig(**config.chronos_config);value.tokenizer_class='StanNormalizeUniformBins'
    return value.create_tokenizer()


def build(stage,meta,device,alignment=None,tiny_contract=False):
    from transformers import AutoTokenizer
    from sensorllm.model import SensorLLMStage1Config,SensorLLMStage2Config,SensorLLMStage1LlamaForCausalLM,SensorLLMStage2LlamaForSequenceClassification
    base=ASSETS/'llama3';chrono=ASSETS/'chronos'
    if not tiny_contract and (not (base/'ASSETS.json').exists() or not (chrono/'ASSETS.json').exists()):
        raise FileNotFoundError('Verified original-equivalent LLaMA-3-8B and author Chronos-T5-large are required')
    tokenizer=AutoTokenizer.from_pretrained(str(alignment/'tokenizer' if alignment else base),local_files_only=True,model_max_length=8192)
    cls,config_cls=(SensorLLMStage1LlamaForCausalLM,SensorLLMStage1Config) if stage==1 else (SensorLLMStage2LlamaForSequenceClassification,SensorLLMStage2Config)
    if tiny_contract:
        config=config_cls(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,num_labels=len(meta.get('classes',[])) or 2,max_position_embeddings=8192)
        # Tiny dimensions exercise plumbing only, never produce benchmark scores.
        model=cls(config).to(device)
    else:
        config=config_cls.from_pretrained(str(base),local_files_only=True)
        config.num_labels=len(meta.get('classes',[])) or 2;config.use_cache=False
        model=cls.from_pretrained(str(base),config=config,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    model.requires_grad_(False);model.get_model().ts_proj.requires_grad_(True)
    model.get_model().fix_llm=True;model.get_model().fix_ts_encoder=True
    # Initialize ALL added input rows before loading an alignment's saved rows.
    base_rows=model.get_input_embeddings().weight.shape[0]
    if alignment:
        state=torch.load(alignment/'alignment.pt',map_location='cpu',weights_only=True)
        model.resize_token_embeddings(state['embedding_rows_end'])
        model.get_model().ts_proj.load_state_dict(state['projector'],strict=True)
        with torch.no_grad():model.get_input_embeddings().weight[state['base_rows']:state['embedding_rows_end']].copy_(state['extra_embeddings'].to(device))
        base_rows=state['base_rows']
    entry=install_schema(model,tokenizer,meta,device)
    model.config.pad_token_id=tokenizer.pad_token_id
    if stage==2:model.score.requires_grad_(True)
    if not tiny_contract:
        model.get_model().load_pt_encoder_backbone_checkpoint(str(chrono),tc='StanNormalizeUniformBins',torch_dtype=torch.bfloat16)
        model.get_model().pt_encoder_backbone.requires_grad_(False)
        model.gradient_checkpointing_enable();model.enable_input_require_grads()
    model._base_rows=base_rows
    return model,tokenizer,entry


def prefix(entry,length):
    tokens=[]
    for key,value in entry.items():
        if key.startswith('default_') and key.endswith('_start_token'):
            end=entry[key.replace('_start_token','_end_token')]
            tokens.append(value+'<ts>'*(length+1)+end+'\n')
    return ''.join(tokens)


def classifier_input(data,meta,entry,tokenizer,chronos):
    from sensorllm.data.utils import preprocess_cls
    summary='Statistics for each channel:\n'+''.join(f'{name}: Mean={float(data[:,i].mean()):.6g}, StdDev={float(data[:,i].std()):.6g}\n' for i,name in enumerate(meta['channels']))
    question='Identify the human activity from: '+', '.join(meta['classes'])+'.'
    # These text fields depend only on the signal and label vocabulary.
    # The sample's true label is deliberately absent from this function.
    result=preprocess_cls([dict(smry=summary,Q=question)],tokenizer,prefix(entry,len(data)),'smry+Q')
    ids,mask,state=chronos.context_input_transform(torch.tensor(data.T,dtype=torch.float64))
    input_ids=result['input_ids'][0][None]
    expected=len(data.T)*(len(data)+1)
    if int((input_ids==tokenizer.convert_tokens_to_ids('<ts>')).sum())!=expected:raise ValueError('SensorLLM prompt truncated time-series tokens')
    return dict(input_ids=input_ids,attention_mask=input_ids.ne(tokenizer.pad_token_id),mts_token_ids=ids[None],mts_attention_mask=mask[None],mts_tokenizer_state=[state])


def compact(model):
    start=model._base_rows;end=model.get_input_embeddings().weight.shape[0]
    result=dict(base_rows=start,embedding_rows_end=end,extra_embeddings=model.get_input_embeddings().weight[start:].detach().cpu(),
                projector={k:v.detach().cpu() for k,v in model.get_model().ts_proj.state_dict().items()})
    if hasattr(model,'score'):result['score']={k:v.detach().cpu() for k,v in model.score.state_dict().items()}
    return result


def contract_matrix():
    from api_adapters import atomic_json
    from transformers import AutoTokenizer
    from unittest.mock import patch
    from sensorllm.model.chronos_model import ChronosConfig
    import sensorllm.model.utils as utilities
    import yaml
    # Use the real tokenizer, native StanNormalize quantizer, original model
    # forward, plus an explicit dummy encoder only for the CPU plumbing fixture.
    cfg=yaml.safe_load((SOURCE/'sensorllm/model/ts_backbone.yaml').read_text())
    cfg['chronos_model'].update(encoder_output_dim=8,projection_hidden_dim=[16,24])
    class Encoder(torch.nn.Module):
        def encode(self,input_ids,attention_mask):return F.one_hot(input_ids%8,8).float()
    import torch.nn.functional as F
    quant=ChronosConfig(tokenizer_class='StanNormalizeUniformBins',tokenizer_kwargs=dict(low_limit=-15.,high_limit=15.),n_tokens=4096,n_special_tokens=2,pad_token_id=0,eos_token_id=1,use_eos_token=True,model_type='seq2seq',context_length=512,prediction_length=64,num_samples=20,temperature=1.,top_k=50,top_p=1.).create_tokenizer()
    rows=[]
    for dataset in ['uci_har','pamap2','usc_had','realworld_har']:
        meta=metadata(dataset);x=np.load(HERE/'data'/dataset/'x.npy',mmap_mode='r')
        sample=input_windows(np.asarray(x[:1]),dataset)[0];sample=np.nan_to_num(sample)
        with patch.object(utilities,'cfg_from_yaml_file',lambda _:copy.deepcopy(cfg)):
            model,tok,entry=build(2,meta,'cpu',tiny_contract=True)
        model.get_model().pt_encoder_backbone=Encoder();model.eval()
        batch=classifier_input(sample,meta,entry,tok,quant)
        with torch.no_grad():output=model(**batch).logits
        assert output.shape==(1,len(meta['classes'])) and torch.isfinite(output).all()
        state=compact(model)
        rows.append(dict(dataset=dataset,channels=sample.shape[1],time_steps=sample.shape[0],input_tokens=batch['input_ids'].numel(),logits_shape=list(output.shape),test_labels_in_prompt=False,compact_extra_rows=state['extra_embeddings'].shape[0]))
        print('SENSOR CONTRACT',rows[-1],flush=True)
    atomic_json(HERE/'audits/sensorllm_contract_20261113.json',dict(status='contract_verified',benchmark_result=False,fixture='Tiny random LLaMA dimensions and dummy TS encoder; genuine LLaMA tokenizer and original quantizer/model forward. No score.',results=rows))


def align(args):
    from api_adapters import atomic_json
    from sensorllm.data.utils import preprocess
    from train import require_finite
    p=ASSETS/'mhealth'
    if not (p/'ASSETS.json').exists():raise FileNotFoundError('Independent official MHEALTH corpus required')
    meta=dict(sampling_rate=50,channels=[f'{place}_{sensor}_{axis}' for place,sensors in [('chest',['acc']),('left_ankle',['acc','gyro']),('right_lower_arm',['acc','gyro'])] for sensor in sensors for axis in 'xyz'])
    meta['classes']=[];device=f'cuda:{args.gpu}'
    model,tok,entry=build(1,meta,device);chronos=chronos_tokenizer();qa=original_qa_functions()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002,weight_decay=0.)
    out=HERE/'sensorllm_alignment_20261113';out.mkdir(exist_ok=True)
    rng=np.random.default_rng(42);steps=0;losses=[]
    # Subject 10 is held out from alignment. No target-benchmark subjects or
    # activity labels are used to generate the alignment targets.
    for file in sorted(p.glob('mHealth_subject*.log')):
        if file.stem=='mHealth_subject10':continue
        raw=np.loadtxt(file,dtype='float32');values=raw[:,[0,1,2,5,6,7,8,9,10,14,15,16,17,18,19]]
        labels=raw[:,-1];bounds=np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1,len(raw)]
        windows=[values[start:start+100] for begin,end in zip(bounds[:-1],bounds[1:]) if labels[begin]>0 for start in range(begin,end-99,100)]
        for wi in rng.permutation(len(windows)):
            window=windows[int(wi)];window=(window-window.mean(0))/np.maximum(window.std(0),1e-6)
            for ch in rng.permutation(15):
                reading=window[:,ch];frame=qa['merge_adjacent_rows'](qa['analyze_trend'](reading,50));pair=qa['select_random_pair']()
                question=qa['dscb_simple_trend'](frame,meta['channels'][ch],pair)
                summary=qa['QA_summary'](reading,frame,meta['channels'][ch],pair)
                starts=[v for k,v in entry.items() if k.endswith('_start_token')];ends=[v for k,v in entry.items() if k.endswith('_end_token')]
                source=dict(Q=starts[ch]+'<ts>'*101+ends[ch]+question['Q'],A=question['A']+'\n\n'+summary['A'])
                batch=preprocess([source],tok,'Analyze the sensor time series and describe its trends.','train','Q','Q')
                ids,mask,state=chronos.context_input_transform(torch.tensor(reading[None],dtype=torch.float64))
                inputs=dict(input_ids=batch['input_ids'][0][None].to(device),labels=batch['labels'][0][None].to(device),ts_token_ids=[ids.to(device)],ts_attention_mask=[mask.to(device)],ts_tokenizer_state=[state],use_cache=False)
                # Stage1 source expects [B,L_ts] for its one-channel encoder.
                optimizer.zero_grad(set_to_none=True);model.train();loss=model(**inputs).loss
                require_finite(loss,'SensorLLM stage1 loss');loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss));steps+=1
                if steps%100==0:print('ALIGN',steps,float(loss),flush=True)
                if args.max_steps and steps>=args.max_steps:break
            if args.max_steps and steps>=args.max_steps:break
        if args.max_steps and steps>=args.max_steps:break
    torch.save(compact(model),out/('alignment_smoke.pt' if args.max_steps else 'alignment.pt'));tok.save_pretrained(out/'tokenizer')
    atomic_json(out/('smoke_result.json' if args.max_steps else 'result.json'),dict(status='smoke_complete' if args.max_steps else 'complete_alignment',steps=steps,last_loss=losses[-1],
        corpus='Official independent MHEALTH subjects1–9, non-null contiguous 100-step windows, all15 channels; subject10 excluded.',
        epochs=1,qa_source='Original notebook trend and summary functions, no API calls',paid_api_calls=0))


def fit_classifier(args):
    from api_adapters import load_fold,prepare_inputs,atomic_json
    from train import metrics,require_finite
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    alignment=HERE/'sensorllm_alignment_20261113'
    if not (alignment/'result.json').exists() or json.loads((alignment/'result.json').read_text())['status']!='complete_alignment':
        raise FileNotFoundError('A complete genuine Stage1 alignment is required before classifier training')
    device=f'cuda:{args.gpu}';meta=metadata(args.dataset)
    x,y,tr,va,te=load_fold(args.dataset,args.seed,args.fold)
    _,normalized,scaler=prepare_inputs(x,tr);x=input_windows(normalized,args.dataset)
    if args.smoke:
        tr=np.concatenate([tr[y[tr]==label][:1] for label in range(len(meta['classes']))])
        va=np.concatenate([va[y[va]==label][:1] for label in range(len(meta['classes']))])
        te=te[:16]
    out=HERE/('smoke_large' if args.smoke else 'large_runs')/'sensorllm'/args.dataset/f'seed_{args.seed}'/args.fold
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    np.savez_compressed(out/'split.npz',train=tr,validation=va,test=te)
    np.savez_compressed(out/'scaler.npz',**{k:np.asarray(v) for k,v in scaler.items()})
    model,tok,entry=build(2,meta,device,alignment);quant=chronos_tokenizer()
    counts=np.bincount(y[tr],minlength=len(meta['classes']));weights=torch.tensor(1/np.maximum(counts,1),device=device,dtype=torch.float32)
    # With one example per microbatch, a weighted mean cancels the class
    # weight. Retain each sample's weight in its gradient contribution.
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights/weights.mean(),reduction='sum')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002,weight_decay=0.)
    def inputs(index):
        batch=classifier_input(x[int(index)],meta,entry,tok,quant)
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    def predict(indices):
        model.eval();values=[]
        with torch.no_grad():
            for index in indices:
                value=model(**inputs(index)).logits.float();require_finite(value,'SensorLLM class logits');values.append(value.cpu().numpy()[0])
        return np.stack(values)
    best=-1;stale=0;history=[]
    for epoch in range(1,(1 if args.smoke else args.epochs)+1):
        order=np.random.permutation(tr);model.train();losses=[]
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            output=model(**inputs(index)).logits.float();loss=loss_fn(output,torch.tensor([int(y[index])],device=device))
            require_finite(loss,'SensorLLM classifier loss');loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
        vl=predict(va);validation=metrics(y[va],vl.argmax(1),len(meta['classes']))
        score=validation['f1_macro' if args.dataset=='realworld_har' else 'accuracy']
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=validation))
        atomic_json(out/'history.json',history)
        if score>best+.0005:best=score;stale=0;torch.save(compact(model),out/'best.pt')
        else:stale+=1
        print('SENSOR CLASSIFIER',args.dataset,args.seed,args.fold,history[-1],flush=True)
        if stale>=10:break
    state=torch.load(out/'best.pt',map_location=device,weights_only=True)
    model.get_model().ts_proj.load_state_dict(state['projector']);model.score.load_state_dict(state['score'])
    with torch.no_grad():model.get_input_embeddings().weight[state['base_rows']:state['embedding_rows_end']].copy_(state['extra_embeddings'])
    logits=predict(te);vl=predict(va)
    np.savez_compressed(out/'predictions.npz',test_indices=te,y_true=y[te],y_pred=logits.argmax(1),logits=logits,validation_indices=va,validation_logits=vl)
    atomic_json(out/'result.json',dict(status='smoke_complete' if args.smoke else 'complete_large_candidate',method='sensorllm',dataset=args.dataset,seed=args.seed,fold=args.fold,
        test=metrics(y[te],logits.argmax(1),len(meta['classes'])),validation=metrics(y[va],vl.argmax(1),len(meta['classes'])),
        counts=dict(train=len(tr),validation=len(va),test=len(te)),normalization=scaler,
        checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),
        recipe=dict(source=json.loads((SOURCE/'SOURCE.json').read_text()),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            stage1='Independent MHEALTH original trend-summary alignment, shared frozen base',stage2='Author SequenceClassification, fixed LLaMA/Chronos, trained original projector, new token embeddings and score head',
            input='UCI total acceleration + gyro6; PAMAP all27 resampled50Hz; USC6 native100Hz; RealWorld42 native50Hz',
            text='Original smry+Q preprocessing; deterministic mean/std summaries extended to native physical channels, no ground-truth activity in input',
            epochs=epoch,calibration='pending independent paper protocol comparison',paid_api_calls=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--contract-matrix',action='store_true');parser.add_argument('--align',action='store_true');parser.add_argument('--gpu',type=int,default=1);parser.add_argument('--max-steps',type=int,default=0)
    parser.add_argument('--dataset');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--fold');parser.add_argument('--smoke',action='store_true');parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    if args.contract_matrix:contract_matrix()
    else:
        torch.cuda.set_device(args.gpu);torch.cuda.set_per_process_memory_fraction(.85,args.gpu)
        if args.align:align(args)
        elif args.dataset and args.fold:fit_classifier(args)
        else:parser.error('Choose --contract-matrix, --align, or --dataset with --fold')
