"""Read-only correctness audit. A passing artifact audit is not paper equivalence.

Run with the experiment Python on server21. Writes only correctness_audit.json;
does not modify data, splits, running jobs, checkpoints or leaderboard approval.
"""
import ast
import collections
import datetime
import hashlib
import json
import numpy as np
from prepare import HERE, SEEDS, atomic_json, sha


def uci_validation_overlap():
    x=np.load(HERE/'data/uci_har/x.npy',mmap_mode='r')
    subjects=np.load(HERE/'data/uci_har/subjects.npy')
    # Direct signal equality, not an assumption that adjacent array indices
    # necessarily came from the same recording. All nine channels must match.
    adjacent=[i for i in range(7351) if subjects[i]==subjects[i+1] and
              np.allclose(x[i,64:],x[i+1,:64],rtol=1e-5,atol=1e-6)]
    rows=[]
    for seed in SEEDS:
        split=np.load(HERE/f'manifests/splits/uci_har/seed_{seed}/official.npz')
        tr,va=set(split['train']),set(split['validation']);touch=set()
        for i in adjacent:
            if i in va and i+1 in tr:touch.add(i)
            if i+1 in va and i in tr:touch.add(i+1)
        assert not set(subjects[split['train']])&set(subjects[split['test']])
        rows.append({'seed':seed,'validation_windows':len(va),
                     'validation_windows_sharing_64_samples_with_train':len(touch),
                     'fraction':len(touch)/len(va),'train_test_subject_overlap':0})
    return {'verified_adjacent_half_window_pairs':len(adjacent),'seeds':rows,
            'interpretation':'Training and validation indices are disjoint but raw samples overlap. Official test subjects remain independent. Validation optimism cannot be equated to a known inflation in held-out test accuracy.'}


def main():
    previous=json.loads((HERE/'verification.json').read_text())
    report=json.loads((HERE/'report.json').read_text())
    runs=[(f,json.loads(f.read_text())) for f in sorted((HERE/'runs').glob('*/*/seed_*/*/result.json'))]
    by_method=collections.defaultdict(list)
    for f,row in runs:by_method[row['method']].append((f,row))
    budgets=[]
    for f,row in by_method['limu_bert']:
        if row['dataset']!='uci_har':continue
        history=json.loads((f.parent/'pretrain_history.json').read_text())
        best=min(history,key=lambda r:r['validation_loss'])
        budgets.append({'seed':row['seed'],'epochs':len(history),'minimum_validation_loss_epoch':best['epoch']})
    semantic_variants={}
    for method,file,names in [('unimts','unimts_adapter.py',{'adapt','FineTune'}),
                              ('limu_bert','limu.py',{'FrozenLimuGRU'}),
                              ('robusthar','robusthar.py',{'RobustHAR'}),
                              ('deepsense','deepsense.py',{'TFGRU','DeepSense'})]:
        variants=collections.Counter()
        for f,row in by_method[method]:
            src=f.parent/'execution_source'/file
            if not src.exists():continue
            tree=ast.parse(src.read_text())
            text='\n'.join(ast.dump(n,include_attributes=False) for n in tree.body if getattr(n,'name',None) in names)
            variants[hashlib.sha256(text.encode()).hexdigest()]+=1
        semantic_variants[method]=dict(variants)
    def released_pooling(batch):
        values=np.ones((batch,2,3,4))
        weights=np.broadcast_to(np.arange(1,3)[None,:,None],(batch,2,1))
        return float(((values*weights[:,:,:,None]).sum(1,keepdims=True)/weights.sum())[0,0,0,0])
    result={
        'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Eight runnable IMU/HAR methods; completed raw-v2 runs. oneHAR released-checkpoint evaluation is separate; MOSAIC has no new run.',
        'overall_verdict':'Held-out scores are reproducible measurements of the declared raw-v2 adapters. Strict original-paper reproduction is not established for this suite.',
        'artifact_verification':{'status':previous['status'],'completed_runs':len(previous['verified_completed_runs']),
            'datasets':len(previous['datasets']),'splits':sum(d['verified_splits'] for d in previous['datasets'].values()),
            'evidence_sha256':sha(HERE/'verification.json'),
            'limits':'Checks saved predictions/metrics/hashes and train-only statistics. Does not prove architecture or protocol equivalence to a paper.'},
        'uci_validation_overlap':uci_validation_overlap(),
        'limu_uci_pretraining':{'author_config_max_epochs':3200,'raw_v2_max_epochs':200,'runs':budgets,
            'inference':'A minimum at epoch199/200 suggests truncation; it does not prove that extending training will improve held-out accuracy.'},
        'executed_core_semantic_variants':semantic_variants,
        'supplementary_probe_log_sha256':sha(HERE/'logs/correctness_probe.log'),
        'onehar_released_pooling_probe':{'same_unit_feature_batch1':released_pooling(1),
            'same_unit_feature_batch2':released_pooling(2),
            'interpretation':'Direct operator probe of author HARmodel weights.sum() across batch and sensors. Not a measured difference in final model predictions, because subsequent normalization can attenuate it.'},
        'sources':{m:{'paper':v['paper'],'repo':v['repo']} for m,v in json.loads((HERE/'methods.json').read_text()).items()},
        'completed_combinations':[{k:c[k] for k in ['method','dataset','metrics','completed_folds']} for c in report['cells'] if c['status']=='complete'],
        'findings':[
            {'id':'A1','priority':'high','type':'validation_independence','methods':'all UCI-HAR runs',
             'finding':'Approximately97-98% of validation windows share64 raw samples with a training window.',
             'code':'prepare.py:205','action':'Create a new version with validation subjects held out inside official training subjects, or recording blocks with a purged gap. Keep official test subjects unchanged and preserve raw-v2 scores as history.'},
            {'id':'A2','priority':'high','type':'method_input_deviation','methods':['unimts'],
             'finding':'Native channels are z-scored before adaptation. Author UCI loader converts acceleration from g to m/s^2; the current path omits that physical-unit contract. OPPORTUNITY BACK uses joint11 while the author mapping uses10.',
             'code':'unimts_adapter.py:17;train.py:155;source/unimts/data.py',
             'action':'Calibrate using the author physical-unit preprocessing and prepared split before accepting the native-window extension. These are valid measurements of the present adapter, not proof of paper equivalence.'},
            {'id':'A3','priority':'high','type':'training_budget_and_task_deviation','methods':['limu_bert'],
             'finding':'Author pretrain config3200 epochs and GRU config700 differ from the current200/200 caps plus early stopping. Native channels/window lengths, normalization and full-label window-level aggregation also differ from the paper low-label protocol.',
             'code':'limu.py:59;source/LIMU-BERT-Public/config/pretrain.json;source/LIMU-BERT-Public/config/train.json',
             'action':'Run an original six-channel/120-step paper-protocol calibration, then separately verify the native-window adapter. Do not tune on existing test scores.'},
            {'id':'A4','priority':'medium','type':'paper_recipe_deviation','methods':['tinyhar','attend_and_discriminate'],
             'finding':'Current batch size128 differs from author/historical calibration batch256. Generic200-epoch budget, validation-primary early stopping and gradient clipping1 also differ from original loops.',
             'code':'train.py:49;train.py:67;train.py:162;train.py:239',
             'action':'Label as canonical recipe variants. Calibrate with original recipes and explicitly bind the intended canonical recipe before claiming a faithful rerun.'},
            {'id':'A5','priority':'high','type':'paper_protocol_and_implementation_uncertainty','methods':['robusthar'],
             'finding':'Current full-label LOSO/113-channel OPPORTUNITY differs from the paper15%-label/random8:1:1/five-device experiment. Width, tokenization, position matrix, norm axis and feature reduction are interpretations because the paper/release does not resolve them.',
             'code':'ROBUSTHAR_AUDIT.md;robusthar.py',
             'action':'Retain as a documented paper-inspired reimplementation pending same-protocol calibration. The48.82 weighted-F1 result cannot be compared directly with paper93.74.'},
            {'id':'A6','priority':'medium','type':'method_adapter_equivalence_pending','methods':['if_convtransformer','deepsense','deepconvlstm'],
             'finding':'IF uses window-local gravity separation and explicit sensor subsets; DeepSense changes temporal FFT blocks and framework; DeepConvLSTM is a peephole/optimizer port. Source arithmetic checks do not establish full original-loader/recipe or end-to-end checkpoint equivalence.',
             'code':'if_adapter.py;deepsense.py;deepconvlstm.py;verify_math.py',
             'action':'Keep these results pending paper-protocol and, for ports, end-to-end numerical calibration.'},
            {'id':'A7','priority':'medium','type':'calibration_evidence_and_gate_limits','methods':['tinyhar'],
             'finding':'Two historical paper-protocol JSON summaries are close to reported paper means, but original score.txt logs are absent locally and on21. Standard promotion currently checks model/commit/dataset hashes without binding the adapter, full recipe and runner fingerprints.',
             'code':'calibration_policy.json;publish_report.py:72',
             'action':'Treat TinyHAR as the strongest, but conditional, evidence. Recover/repeat original calibrations and extend fingerprint checks before broad automatic promotion.'},
            {'id':'A8','priority':'medium','type':'window_label_protocol_choice','methods':'all OPPORTUNITY raw-v2 runs',
             'finding':'Non-null-only majority can label windows dominated by NULL. A separate raw-data probe found1620/8886 windows with fewer than15/30 non-null frames and1643 with no class occupying half of all frames; minimum non-null support is1 frame.',
             'code':'prepare.py:171','action':'Freeze and describe this rule as a specific protocol choice. Do not equate its task/results with a paper using last-frame, all-frame-majority, NULL-inclusive or different sensor protocols.'},
            {'id':'A9','priority':'medium','type':'released_checkpoint_evaluation_only','methods':['onehar'],
             'finding':'Author released-checkpoint evaluation is available and uses strictly loaded weights. Its loader estimates position statistics from the full evaluation set; HARmodel pooling divides by a weights.sum() that includes the batch dimension. Original training membership is not established by released weights.',
             'code':'source/uniHAR/test.py:70;source/uniHAR/HARmodel.py:62',
             'action':'Keep as author-release inference evidence with exact batching/normalization, separate from clean retraining and subject-held-out Standard measurements. Do not patch these operations and silently reuse the old checkpoint scores.'}
        ],
        'method_assessment':{
            'tinyhar':'Official architecture unchanged; historical summary calibration supports plausibility; UCI validation overlap and changed canonical recipe remain limitations.',
            'attend_and_discriminate':'Official architecture/mixup/center loss matched; source recipe extended to UCI/WISDM with changed budget/batch/clipping. Pending calibration.',
            'if_convtransformer':'Official model arithmetic matched; physical-input/preprocessing extension. Pending calibration.',
            'unimts':'Strict-loaded author backbone and correct classifier; physical-input and BACK-joint deviations require a calibrated adapter.',
            'limu_bert':'Official encoder/masker/GRU used; major budget, native-input and evaluation-granularity changes. Under-convergence is plausible, not proven.',
            'deepsense':'Core TF1 reset-before GRU, channel-wise dropout and source FFT layout were inspected; full port equivalence and time-scale calibration remain unproven.',
            'deepconvlstm':'Peephole recurrence and Lasagne RMSprop math checked; author checkpoint end-to-end equivalence and paper-protocol calibration remain unproven.',
            'robusthar':'Corrected formula wiring passed internal checks. Important paper underspecification and evaluation differences prevent a strict reproduction claim.'},
        'training_or_approval_modified_by_this_audit':False,
    }
    atomic_json(HERE/'correctness_audit.json',result)
    print(json.dumps({'saved':'correctness_audit.json','verified_completed_runs':result['artifact_verification']['completed_runs'],
        'overlap':[r['validation_windows_sharing_64_samples_with_train'] for r in result['uci_validation_overlap']['seeds']],
        'findings':len(result['findings']),'verdict':result['overall_verdict']}))


if __name__=='__main__':main()
