"""Offline-first HARGPT / author ByMyEyes adapters. No network in dry-run mode.

Setups and fixtures are stored outside scored runs. A test label is never passed
to a prompt builder. Only a future, explicit --execute plus an authorized
configuration can construct a network client.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np
from scipy.signal import resample_poly

HERE=Path(__file__).resolve().parent


class PaidCallsDisabled(RuntimeError):pass


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class GuardedChat:
    """Original Chat Completions content schema, persistent call-limit guard.

    Deliberately no SDK retries: a timeout may already be billable. A reserved
    call is retained on failure. Budget values must be explicitly configured;
    a conservative per-request reservation is required before enabling calls.
    """
    def __init__(self,config,cache,execute=False):
        self.config=config;self.cache=Path(cache);self.execute=execute
        self.requests=[];self.cache.mkdir(parents=True,exist_ok=True)

    def generate(self,content):
        payload=dict(model=self.config.get('model'),messages=[dict(role='user',content=content)],
                     temperature=0,max_tokens=int(self.config.get('max_output_tokens',512)))
        key=fingerprint(payload);self.requests.append(dict(key=key,payload=payload))
        if not self.execute or self.config.get('allow_paid_calls') is not True:
            raise PaidCallsDisabled('Paid calls are disabled by the user; payload generated locally only.')
        if not payload['model']:raise ValueError('Explicit model/snapshot required; no automatic replacement.')
        saved=self.cache/f'{key}.json'
        if saved.exists():return json.loads(saved.read_text())['text']
        maximum=int(self.config.get('max_calls',0));budget=float(self.config.get('budget_usd',0))
        reservation=float(self.config.get('max_request_cost_usd',0))
        if maximum<=0 or not 0<reservation<=budget:raise ValueError('Explicit call limit and positive budget/reservation required.')
        # The reservation must include all text and image input plus the output
        # cap for the exact configured model. Never infer contemporary pricing.
        if not self.config.get('request_cost_bound_verified'):raise ValueError('A verified per-request cost upper bound is required.')
        credential=os.environ.get(self.config.get('api_key_env','OPENAI_API_KEY'))
        if not credential:raise ValueError('Configured API key environment variable is absent.')
        db=sqlite3.connect(self.cache/'budget.sqlite',timeout=30,isolation_level=None)
        db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, key TEXT, reserved REAL, status TEXT)')
        db.execute('BEGIN IMMEDIATE')
        n,spent=db.execute('SELECT COUNT(*),COALESCE(SUM(reserved),0) FROM calls').fetchone()
        if n>=maximum or spent+reservation>budget:
            db.rollback();db.close();raise PaidCallsDisabled('Configured call/budget limit reached.')
        cursor=db.execute('INSERT INTO calls(key,reserved,status) VALUES (?,?,?)',(key,reservation,'reserved'))
        call_id=cursor.lastrowid;db.commit()
        try:
            from openai import OpenAI
            client=OpenAI(api_key=credential,base_url=self.config.get('base_url','https://api.openai.com/v1'),
                          timeout=120,max_retries=0)
            response=client.chat.completions.create(**payload)
            text=response.choices[0].message.content
            if not isinstance(text,str):raise ValueError('No textual model answer')
            atomic_json(saved,dict(text=text,request_sha256=key,response=response.model_dump(mode='json')))
            db.execute('UPDATE calls SET status=? WHERE id=?',('complete',call_id))
            return text
        finally:db.close()


class FixtureChat:
    """Deterministic offline contract fixture; never a measured prediction."""
    def __init__(self,answers):self.answers=iter(answers);self.requests=[]
    def generate(self,content):
        self.requests.append(content)
        return next(self.answers)


def dataset_metadata(dataset):
    manifest=json.loads((HERE/'data'/dataset/'manifest.json').read_text())
    sr={'uci_har':50,'pamap2':100,'usc_had':100,'realworld_har':50}[dataset]
    channels=[]
    if dataset=='uci_har':channels=[f'{sensor}_{axis}' for sensor in ['body_acc','body_gyro','total_acc'] for axis in 'xyz']
    if dataset=='pamap2':channels=[f'{position}_{sensor}_{axis}' for position in ['hand','chest','ankle'] for sensor in ['acc','gyro','mag'] for axis in 'xyz']
    if dataset=='usc_had':channels=[f'hip_{sensor}_{axis}' for sensor in ['acc','gyro'] for axis in 'xyz']
    if dataset=='realworld_har':channels=[f'{position}_{sensor}_{axis}' for position in ['chest','forearm','head','shin','thigh','upperarm','waist'] for sensor in ['acc','gyro'] for axis in 'xyz']
    assert len(channels)==manifest['shape'][2]
    return dict(classes=manifest['classes'],sampling_rate=sr,channels=channels,
                task_description='recognizing human activities',
                data_description=f'{dataset} wearable inertial sensor window. Channel names identify position, sensor and axis. Values use training-fold median imputation; visual inputs additionally use training-fold z-score.')


def hargpt_prompt(data,metadata):
    sr=metadata['sampling_rate'];reduced=resample_poly(data,10,sr,axis=0)
    signals={name:np.round(reduced[:,i],3).tolist() for i,name in enumerate(metadata['channels'])}
    instruction='You are an expert in analyzing inertial sensor data for human activity recognition.'
    question=(f"The wearable sensors provide the following raw readings at 10 Hz. "
              f"The possible activities are {json.dumps(metadata['classes'])}. "
              f"Channel names specify the body location, measurement and axis.\n{json.dumps(signals,separators=(',',':'))}\n"
              'Which of the listed activities best describes this window? Please analyze the signal and give the final activity name.')
    return [dict(type='text',text=instruction+'\n'+question)]


def parse_answer(text,classes):
    tagged=re.findall(r'<answer>\s*(.*?)\s*</answer>',text,flags=re.I|re.S)
    candidate=(tagged[-1] if tagged else text.splitlines()[-1]).strip().removeprefix('ANSWER: ').strip(' ."\'')
    normalized=lambda v:re.sub(r'[^a-z0-9]+',' ',v.lower()).strip()
    matches=[i for i,name in enumerate(classes) if normalized(candidate)==normalized(name)]
    if len(matches)!=1:raise ValueError(f'Unparseable/ambiguous answer: {candidate[:150]}')
    return matches[0]


def bymyeyes_modules():
    directory=HERE/'vendor/bymyeyes';sys.path.insert(0,str(directory))
    from core.solver import Solver
    from core.vis_generator import VisualizationGenerator
    from core.logger import Logger
    return Solver,VisualizationGenerator,Logger


def bymyeyes_setup(llm,metadata,examples,log_dir):
    Solver,Generator,Logger=bymyeyes_modules();logger=Logger(str(log_dir),debug=False)
    generator=Generator(llm,metadata,logger)
    # Original planner + selection, limited to relevant original inertial plots.
    generator.visualizations={k:v for k,v in generator.visualizations.items() if k in ['raw waveform','spectrogram','power spectral density']}
    candidates=generator.plan('planning')
    if not isinstance(candidates,list) or not candidates:raise ValueError('Empty visualization plan')
    for candidate in candidates:
        if candidate['func'] not in generator.visualizations:raise ValueError('Invalid inertial visualization')
    selected=generator.select(candidates,examples,'selection')
    if selected is None:raise ValueError('Visualization selection did not match the plan')
    # Logging token estimate only. The original paper configuration is retained;
    # the actual provider model must be separately specified in api_config.json.
    config=dict(use_vis=True,vis_func=selected['func'],vis_args=dict(selected['args']),llm_version='gpt-4o')
    return Solver(llm,config,metadata,logger),dict(candidates=candidates,selected=selected)


def load_fold(dataset,seed,fold):
    directory=HERE/'data'/dataset
    x=np.load(directory/'x.npy',mmap_mode='r');y=np.load(directory/'y.npy')
    split=np.load(HERE/'manifests/splits'/dataset/f'seed_{seed}'/f'{fold}.npz')
    # Import common train-only scaler, never fit with test or validation data.
    tr,va,te=(split[k] for k in ['train','validation','test'])
    assert not set(tr)&set(te) and not set(va)&set(te) and not set(tr)&set(va)
    return x,y,tr,va,te


def prepare_inputs(x,tr):
    train=np.asarray(x[tr]);median=np.nanmedian(train,axis=(0,1))
    median=np.nan_to_num(median,nan=0.)
    filled=np.where(np.isfinite(x),x,median)
    mean=filled[tr].mean(axis=(0,1));std=filled[tr].std(axis=(0,1));std=np.maximum(std,1e-6)
    return filled.astype('float32'),((filled-mean)/std).astype('float32'),dict(median=median.tolist(),mean=mean.tolist(),std=std.tolist())


def dry_run(dataset,seed,fold):
    x,y,tr,va,te=load_fold(dataset,seed,fold);metadata=dataset_metadata(dataset)
    raw,normalized,scaler=prepare_inputs(x,tr)
    output=HERE/'api_dry_runs_20261113'/dataset/f'seed_{seed}'/fold
    output.mkdir(parents=True,exist_ok=True)
    examples_ids=[]
    for label in range(len(metadata['classes'])):
        available=tr[y[tr]==label]
        if len(available):examples_ids.append(int(available[0]))
    assert set(examples_ids)<=set(map(int,tr))
    examples=[(normalized[i],metadata['classes'][int(y[i])]) for i in examples_ids]
    candidates=[dict(func='raw waveform',args={},knowledge='Offline fixture, not an LLM-produced plan.')]
    fixture=FixtureChat([json.dumps(candidates),json.dumps(dict(func='raw waveform')),
                         '<answer>'+metadata['classes'][0]+'</answer>'])
    solver,selection=bymyeyes_setup(fixture,metadata,examples,output/'bymyeyes')
    answer=solver.solve(normalized[int(te[0])],examples,'target')
    assert parse_answer(answer,metadata['classes'])==0
    content=hargpt_prompt(raw[int(te[0])],metadata)
    guard=GuardedChat(dict(allow_paid_calls=False,model=None),output/'disabled_cache',execute=True)
    try:guard.generate(content)
    except PaidCallsDisabled:pass
    else:raise AssertionError('Disabled client must not execute')
    assert len(list((output/'disabled_cache').glob('*.json')))==0
    atomic_json(output/'requests.json',dict(status='offline_contract_verified',not_a_benchmark_result=True,
        dataset=dataset,seed=seed,fold=fold,test_window_index=int(te[0]),training_example_indices=examples_ids,
        train_test_disjoint=True,hargpt=content,bymyeyes=fixture.requests,selection=selection,
        scaler=scaler,paid_api_calls=0,parse_errors_are_not_scored=True,
        hargpt_implementation='Paper-guided raw-10Hz prompt adapter; not original source or exact prompt reproduction.',
        bymyeyes_implementation='Original planning, selection, solver and visualization code; fixture answers used only to validate interface.'))
    return dict(dataset=dataset,status='offline_contract_verified',paid_api_calls=0,path=str(output/'requests.json'))


def evaluate(method,dataset,seed,fold,config,execute=False):
    if not execute or config.get('allow_paid_calls') is not True:raise PaidCallsDisabled('User authorized interfaces only; evaluation cannot run.')
    x,y,tr,va,te=load_fold(dataset,seed,fold);metadata=dataset_metadata(dataset)
    raw,normalized,scaler=prepare_inputs(x,tr)
    out=HERE/'api_evaluations'/method/dataset/f'seed_{seed}'/fold
    llm=GuardedChat(config,HERE/'api_cache'/method,execute=True)
    examples=[];ids=[];solver=None
    if method=='bymyeyes':
        rng=np.random.default_rng(seed)
        for label,name in enumerate(metadata['classes']):
            available=tr[y[tr]==label]
            if len(available):
                idx=int(rng.choice(available));ids.append(idx);examples.append((normalized[idx],name))
        solver,selection=bymyeyes_setup(llm,metadata,examples,out/'visualization')
        atomic_json(out/'selection.json',selection)
    predictions=[]
    for index in te:
        response=llm.generate(hargpt_prompt(raw[int(index)],metadata)) if method=='hargpt' else solver.solve(normalized[int(index)],examples,f'target_{int(index)}')
        try:pred=parse_answer(response,metadata['classes']);error=None
        except ValueError as exc:pred=-1;error=str(exc)
        predictions.append(dict(index=int(index),prediction=pred,parse_error=error,response=response))
        atomic_json(out/'predictions.json',predictions)
    # Parse failures remain in the denominator as wrong predictions.
    from sklearn.metrics import accuracy_score,f1_score
    predicted=np.array([p['prediction'] for p in predictions])
    result=dict(status='complete_api_evaluation',method=method,dataset=dataset,seed=seed,fold=fold,
        test_count=len(te),training_example_indices=ids,parse_failures=int((predicted<0).sum()),
        test=dict(accuracy=accuracy_score(y[te],predicted),f1_macro=f1_score(y[te],predicted,labels=list(range(len(metadata['classes']))),average='macro',zero_division=0)),
        protocol='API candidate, original target split; train-only examples; not inserted into canonical training rank.',
        model=config['model'],scaler=scaler)
    atomic_json(out/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['hargpt','bymyeyes'])
    parser.add_argument('--dataset');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--fold')
    parser.add_argument('--dry-run-matrix',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--config',type=Path,default=HERE/'api_config_20261113.json');args=parser.parse_args()
    if args.dry_run_matrix:
        rows=[dry_run(ds,42,fold) for ds,fold in [('uci_har','official'),('pamap2','test_101_val_102'),('usc_had','test_1_val_2'),('realworld_har','test_1_val_2')]]
        atomic_json(HERE/'audits/api_adapter_dry_run_20261113.json',dict(paid_api_calls=0,results=rows))
        print(json.dumps(rows,indent=2))
    else:print(evaluate(args.method,args.dataset,args.seed,args.fold,json.loads(args.config.read_text()),args.execute))
