"""Exercise complete train/scaler/checkpoint/prediction flow outside real runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare import HERE,atomic_json

if __name__=='__main__':
    outcomes=[]
    for method in ['tinyhar','attend_and_discriminate','robusthar','limu_bert','deepconvlstm','deepsense','if_convtransformer','unimts']:
        for dataset in ['uci_har','wisdm','pamap2','hhar','opportunity_a']:
            split=sorted((HERE/'manifests/splits'/dataset/'seed_42').glob('*.npz'))[0]
            log=HERE/'logs'/f'smoke__{method}__{dataset}.log'
            cmd=[sys.executable,'-u',str(HERE/'train.py'),'--method',method,'--dataset',dataset,
                 '--fold',split.stem,'--seed','42','--gpu','3','--epochs','1','--smoke',
                 '--batch-size','8','--pretrain-batch-size','8','--memory-fraction','.1']
            with log.open('wb') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
            result=HERE/'smoke'/method/dataset/'seed_42'/split.stem/'result.json'
            passed=code==0 and result.exists() and json.loads(result.read_text())['status']=='smoke_complete'
            outcomes.append({'method':method,'dataset':dataset,'passed':passed,'exit_code':code,'log':str(log.relative_to(HERE))})
            atomic_json(HERE/'smoke_verification.json',outcomes)
            print('PASS' if passed else 'FAIL',method,dataset,flush=True)
    raise SystemExit(0 if all(r['passed'] for r in outcomes) else 1)
