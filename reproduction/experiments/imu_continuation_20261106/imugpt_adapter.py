"""IMUGPT cached-prompt/native-simulation bridge and calibrated HAR transfer.

Synthetic input must come from the genuine text→motion→IMU pipeline and carry
its provenance. Missing assets never fall back to real-only training or random
signals. Cached author descriptions avoid any paid prompt-generation API.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.signal import resample_poly

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'vendor/imugpt'
ASSETS=HERE/'assets_20261113/imugpt'
ALIASES={
 'uci_har':('usc-had',['walking forward','climbing up stairs','climbing down stairs','sitting','standing','lying']),
 'usc_had':('usc-had',['walking forward','walking counter-clockwise','walking clockwise','climbing up stairs','climbing down stairs','running','jumping','sitting','standing','lying','riding up elevator','riding down elevator']),
 'pamap2':('pamap',['Lying','Sitting','Standing','Walking','Running','Cycling','Nordic walking','Ascending stairs','Descending stairs','Vacuum cleaning','Ironing','Rope jumping']),
 'realworld_har':('realworld',['climb down stairs','climb up stairs','jumping','lying','running','sitting','standing','walking']),
}
JOINTS={
 'uci_har':['Right_hip'],'usc_had':['Right_hip'],
 'pamap2':['Right_wrist','Spine3','Right_ankle'],
 'realworld_har':['Spine3','Left_wrist','Head','Left_ankle','Left_knee','Left_elbow','Spine1'],
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_manifest(dataset):
    directory,names=ALIASES[dataset];rows=[]
    for index,name in enumerate(names):
        file=SOURCE/'prompts'/directory/(name+'.txt')
        if not file.exists():
            matches=[p for p in (SOURCE/'prompts'/directory).glob('*.txt') if p.stem.lower()==name.lower()]
            if len(matches)!=1:raise FileNotFoundError(f'No unambiguous author prompt for {dataset}/{name}')
            file=matches[0]
        try:text=file.read_text(encoding='utf-8');encoding='utf-8'
        except UnicodeDecodeError:text=file.read_text(encoding='cp1252');encoding='cp1252'
        lines=[line.strip() for line in text.splitlines() if line.strip()]
        if not lines:raise ValueError(f'Empty author prompt file {file}')
        rows.append(dict(class_id=index,alias=name,prompt_file=str(file),prompt_sha256=sha(file),encoding=encoding,descriptions=lines))
    return rows


def real_channels(x,dataset):
    if dataset=='uci_har':return x[:,:,[6,7,8,3,4,5]]
    if dataset=='pamap2':return x[:,:,np.r_[0:6,9:15,18:24]]
    return x


def calibrated(virtual,real_train):
    file=SOURCE/'check_calibration/ProbabilityIntegralTransfer.py'
    spec=importlib.util.spec_from_file_location('author_imugpt_calibration',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    shape=virtual.shape
    # Original unsupervised percentile transfer; real_train must be the exact
    # current training fold. Never use validation/test distribution or labels.
    result=module.prob_integ_xfer(virtual.reshape(-1,shape[-1]),real_train.reshape(-1,shape[-1]),
                                 perClass=False,resolution=100,num_threads=4,verbose=False)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite author IMUGPT calibration')
    return result.reshape(shape).astype('float32')


def prepare_model(x,train,validation,out,a,device):
    from deepconvlstm import DeepConvLSTM
    from train import require_finite
    data=ASSETS/'synthetic'/f'{a.dataset}.npz';manifest=data.with_suffix('.json')
    if not data.exists() or not manifest.exists():raise FileNotFoundError('Genuine IMUGPT synthetic IMU corpus is not ready; real-only proxy is forbidden')
    info=json.loads(manifest.read_text())
    if info.get('generation_pipeline')!='author_T2M_GPT_BVH_IMUSim':raise ValueError('Synthetic provenance is missing')
    if info['npz_sha256']!=sha(data):raise ValueError('Synthetic corpus hash changed')
    classes=json.loads((HERE/'data'/a.dataset/'manifest.json').read_text())['classes']
    if info['classes']!=classes:raise ValueError('Synthetic/canonical class vocabulary mismatch')
    values=np.load(data);virtual=values['x'];labels=values['y'].astype('int64')
    x=real_channels(x,a.dataset)
    if virtual.shape[1:]!=x.shape[1:]:raise ValueError('Synthetic channels/window differ from the canonical adapter')
    if set(labels)!=set(range(len(classes))):raise ValueError('Every target class needs genuine virtual examples')
    virtual=calibrated(virtual,x[train])
    model=DeepConvLSTM(x.shape[2],len(classes)).to(device)
    optimizer=torch.optim.RMSprop(model.parameters(),lr=.001)
    vx=torch.tensor(virtual,device=device).unsqueeze(1);vy=torch.tensor(labels,device=device)
    history=[]
    for epoch in range(1,(1 if a.smoke else 5)+1):
        order=np.random.permutation(len(labels));model.train();losses=[]
        for start in range(0,len(order),100):
            ids=order[start:start+100];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(vx[ids]),vy[ids]);require_finite(loss,'IMUGPT synthetic pretraining loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    from prepare import atomic_json
    atomic_json(out/'virtual_pretraining.json',dict(epochs=history,synthetic_windows=len(labels),corpus=info,calibration_fit_count=len(train),used_validation=False,used_test=False))
    return model,dict(_input_array=x,input='B,1,T,C',optimizer='RMSprop',lr=.001,batch_size=100,weight_decay=0.,loss='cross_entropy',
        architecture='IMUGPT synthetic pretraining then real-fold fine-tuning with audited DeepConvLSTM peephole port',
        official_repo='https://github.com/ZikangLeng/IMUGPT',official_commit=json.loads((SOURCE/'SOURCE.json').read_text())['commit'],
        implementation_type='official_generation_and_calibration_with_classifier_transfer_adapter',
        model_sha256=sha(HERE/'deepconvlstm.py'),calibration_sha256=sha(SOURCE/'check_calibration/ProbabilityIntegralTransfer.py'),
        adapter_sha256=sha(Path(__file__)),synthetic_sha256=sha(data),virtual_pretrain_epochs=len(history),
        sensor_adapter='UCI total_acc+gyro6; PAMAP18 acc/gyro (no simulated magnetometer); USC hip6; RealWorld42 in canonical physical order.',
        compatibility='UCI label extension reuses matching cached USC descriptions. USC simulated hip matches canonical placement; author notebook instead selected knee. No API prompt generation.',
        calibration='Canonical candidate; synthetic count, five-epoch pretraining and split differ from original paper; paper protocol calibration pending')


def native_motion_models(device):
    from unittest.mock import patch
    import open_clip
    sys.path.insert(0,str(SOURCE))
    import options.option_transformer as options
    import models.vqvae as vqvae
    import models.t2m_trans as trans
    with patch.object(sys,'argv',['imugpt']):args=options.get_args_parser()
    args.dataname='t2m';args.down_t=2;args.depth=3;args.block_size=51
    weights=ASSETS/'pretrained'
    vq=list(weights.rglob('VQVAE/net_last.pth'));gpt=list(weights.rglob('VQTransformer_corruption05/net_best_fid.pth'))
    if len(vq)!=1 or len(gpt)!=1:raise FileNotFoundError('Author VQ-VAE and motion Transformer weights are required')
    model=vqvae.HumanVQVAE(args,args.nb_code,args.code_dim,args.output_emb_width,args.down_t,args.stride_t,args.width,args.depth,args.dilation_growth_rate)
    motion=trans.Text2Motion_Transformer(num_vq=args.nb_code,embed_dim=1024,clip_dim=args.clip_dim,block_size=51,num_layers=9,n_head=16,drop_out_rate=args.drop_out_rate,fc_rate=args.ff_rate)
    model.load_state_dict(torch.load(vq[0],map_location='cpu',weights_only=False)['net'],strict=True)
    motion.load_state_dict(torch.load(gpt[0],map_location='cpu',weights_only=False)['trans'],strict=True)
    clip,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',precision='fp16',device=device,cache_dir=str(ASSETS/'clip'))
    clip=clip.to(device).eval();tokenize=open_clip.get_tokenizer('ViT-B-32')
    normalization=ASSETS/'normalization'
    mean=normalization/'mean.npy';std=normalization/'std.npy'
    manifest=ASSETS/'NORMALIZATION_ASSETS.json'
    if not mean.exists() or not std.exists() or not manifest.exists():raise FileNotFoundError('Exact author T2M normalization arrays missing')
    info=json.loads(manifest.read_text())
    for file in [mean,std]:
        if sha(file)!=info['files'][file.name]['sha256']:raise ValueError('Author motion normalization SHA mismatch')
    return model.to(device).eval(),motion.to(device).eval(),clip,tokenize,torch.tensor(np.load(mean),device=device),torch.tensor(np.load(std),device=device)


def generate(args):
    from api_adapters import dataset_metadata,atomic_json
    torch.cuda.set_device(args.gpu);torch.cuda.set_per_process_memory_fraction(.85,args.gpu)
    sys.path.insert(0,str(SOURCE))
    torch.manual_seed(42);np.random.seed(42)
    # Loading only the author's IK function avoids importing the paid prompt
    # generation module, unrelated OpenAI clients, and visualization notebooks.
    from Motion.InverseKinematics import animation_from_positions
    from Motion import BVH
    from utils.motion_process import recover_from_ric
    tree=ast.parse((SOURCE/'data_generation.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['motion2bvh_ik','rescale_bvh']]
    scope=dict(np=np,animation_from_positions=animation_from_positions,BVH=BVH)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/'data_generation.py'),'exec'),scope)
    sensor_code=(SOURCE/'sensor.py').read_text()
    pattern=r'^  return sensor[ \t]*$'
    assert len(re.findall(pattern,sensor_code,flags=re.M))==1
    sensor_code=re.sub(pattern,"  sensor['sampling_rate_hz'] = 1/samplingPeriod\n  return sensor",sensor_code,flags=re.M)
    from types import ModuleType
    sensor=ModuleType('author_imugpt_sensor');sensor.__file__=str(SOURCE/'sensor.py')
    exec(compile(sensor_code,str(SOURCE/'sensor.py'),'exec'),sensor.__dict__)
    model,motion,clip,tokenize,mean,std=native_motion_models(f'cuda:{args.gpu}')
    meta=dataset_metadata(args.dataset);rate=meta['sampling_rate'];length=json.loads((HERE/'data'/args.dataset/'manifest.json').read_text())['shape'][1]
    work=ASSETS/'generated'/args.dataset;work.mkdir(parents=True,exist_ok=True)
    values=[];labels=[];sources=[]
    for group in prompt_manifest(args.dataset):
        for index,description in enumerate(group['descriptions'][:args.max_prompts]):
            name=f"class_{group['class_id']}_{index}";bvh=work/(name+'.bvh');acc=work/(name+'_acc.npz');gyro=work/(name+'_gyro.npz');clock=work/(name+'_clock.json')
            if not bvh.exists():
                with torch.no_grad():
                    feature=clip.encode_text(tokenize([description]).to(f'cuda:{args.gpu}')).float()
                    codes=motion.sample(feature,False);pose=model.forward_decoder(codes)
                    xyz=recover_from_ric((pose*std+mean).float(),22)[0].cpu().numpy()
                raw_bvh=work/(name+'_raw.bvh')
                scope['motion2bvh_ik'](xyz.copy(),str(raw_bvh))
                scope['rescale_bvh'](str(raw_bvh),str(bvh),1.7/1.55)
            # The native source has no UCI placement; its USC branch generates
            # both right hip and knee, permitting an explicit waist extension.
            mode={'uci_har':'usc-had','usc_had':'usc-had','pamap2':'pamap','realworld_har':'realworld'}[args.dataset]
            if not acc.exists() or not gyro.exists() or not clock.exists():
                sensor_info=sensor.extract_vir_imu(str(bvh),str(acc),str(gyro),mode)
                atomic_json(clock,dict(sampling_rate_hz=sensor_info['sampling_rate_hz']))
            a,g=np.load(acc),np.load(gyro);joints=JOINTS[args.dataset]
            count=min(min(len(a[j]),len(g[j])) for j in joints)
            combined=np.concatenate([np.c_[a[j][:count],g[j][:count]] for j in joints],axis=1)
            # Record the native simulator's exact sampling period. The sole
            # source instrumentation returns this scalar without changing math.
            source_rate=json.loads(clock.read_text())['sampling_rate_hz']
            from fractions import Fraction
            ratio=Fraction(rate/source_rate).limit_denominator(10000)
            combined=resample_poly(combined,ratio.numerator,ratio.denominator,axis=0)
            for start in range(0,len(combined)-length+1,length//2):values.append(combined[start:start+length].astype('float32'));labels.append(group['class_id'])
            sources.append(dict(class_id=group['class_id'],description=description,bvh_sha256=sha(bvh),acc_sha256=sha(acc),gyro_sha256=sha(gyro),source_rate_hz=source_rate))
    if not values:raise RuntimeError('No genuine motion windows generated')
    if set(labels)!=set(range(len(meta['classes']))):raise RuntimeError('Some target classes have no genuine virtual windows; corpus cannot be admitted')
    output=ASSETS/'synthetic'/f'{args.dataset}.npz';output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,x=np.stack(values),y=np.asarray(labels,dtype='int64'))
    atomic_json(output.with_suffix('.json'),dict(generation_pipeline='author_T2M_GPT_BVH_IMUSim',classes=meta['classes'],npz_sha256=sha(output),paid_api_calls=0,
        source=json.loads((SOURCE/'SOURCE.json').read_text()),prompts_per_class=args.max_prompts,sources=sources,joints=JOINTS[args.dataset],input_extension='Canonical sensor/label mapping; no SMPL body files needed by original IK route.'))


def preflight():
    from api_adapters import atomic_json
    rows=[]
    for dataset in ALIASES:
        prompts=prompt_manifest(dataset);rows.append(dict(dataset=dataset,classes=len(prompts),cached_descriptions=sum(len(g['descriptions']) for g in prompts),prompt_files=[dict(class_id=g['class_id'],file=g['prompt_file'],sha256=g['prompt_sha256']) for g in prompts],joints=JOINTS[dataset]))
    rng=np.random.default_rng(42);virtual=rng.normal(size=(4,100,6)).astype('float32');real=rng.normal(loc=2,scale=3,size=(5,100,6)).astype('float32')
    result=calibrated(virtual,real)
    assert result.shape==virtual.shape and np.isfinite(result).all()
    atomic_json(HERE/'audits/imugpt_contract_20261113.json',dict(status='cached_prompt_and_calibration_contract_verified',benchmark_result=False,fixture='Random arrays test original percentile transfer only; never saved as synthetic training data.',results=rows,
        motion_weights_ready=(ASSETS/'MOTION_ASSETS.json').exists(),synthetic_corpora_ready={ds:(ASSETS/'synthetic'/f'{ds}.npz').exists() for ds in ALIASES},paid_api_calls=0))
    print('IMUGPT CONTRACT',[(r['dataset'],r['classes'],r['cached_descriptions']) for r in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--generate',action='store_true');p.add_argument('--dataset',choices=list(ALIASES));p.add_argument('--max-prompts',type=int,default=10);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--train',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--fold');p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=100);p.add_argument('--memory-fraction',type=float,default=.16);p.add_argument('--batch-size',type=int);args=p.parse_args()
    if args.preflight:preflight()
    elif args.generate:generate(args)
    elif args.train:
        from train import run
        args.method='imugpt';run(args)
    else:p.error('Choose --preflight or --generate')
