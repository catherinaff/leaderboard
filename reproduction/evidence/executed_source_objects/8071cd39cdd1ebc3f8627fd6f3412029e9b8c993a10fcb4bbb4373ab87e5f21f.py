"""Independent artifact audit: official archives, subject splits, saved predictions."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from prepare import HERE,SEEDS,sha,atomic_json

EXPECTED={
 'uci_har':'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031',
 'wisdm':'e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7',
 'pamap2':'76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352',
 'hhar':'8c9f8dd5f50074feaad2c5cec5d927a4fc8366324de26079629a3de3aef0e124',
 'opportunity_a':'3df913b5e5a987d309596350385087737784c9d4b811389536eb3a9bcaf6f1d0'}


def main():
    datasets={};runs=[]
    for name,expected in EXPECTED.items():
        base=HERE/'data'/name
        meta=json.loads((base/'manifest.json').read_text())
        assert meta['archive_sha256']==expected,name
        for key,h in meta['arrays_sha256'].items():assert sha(base/f'{key}.npy')==h,(name,key)
        y=np.load(base/'y.npy');subjects=np.load(base/'subjects.npy')
        folds=list((HERE/'manifests/splits'/name).glob('seed_*/*.npz'))
        expected_folds=5*{'uci_har':1,'wisdm':1,'pamap2':8,'hhar':9,'opportunity_a':4}[name]
        assert len(folds)==expected_folds
        for path in folds:
            split=np.load(path);tr,va,te=[split[k] for k in ['train','validation','test']]
            assert len(set(tr)|set(va)|set(te))==len(y)
            assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
            assert not set(subjects[tr])&set(subjects[te])
            if name!='uci_har':
                assert not set(subjects[tr])&set(subjects[va]) and not set(subjects[va])&set(subjects[te])
            if name=='uci_har':
                assert len(tr)+len(va)==7352 and len(te)==2947
                rec=np.load(base/'recordings.npy')
                assert (rec[tr]==0).all() and (rec[va]==0).all() and (rec[te]==1).all()
            if name=='wisdm':
                assert set(subjects[tr])==set(range(1,29)) and set(subjects[va])=={29,30} and set(subjects[te])==set(range(31,37))
        datasets[name]={'manifest_sha256':sha(base/'manifest.json'),'shape':meta['shape'],
                        'official_archive_sha256':expected,'verified_splits':len(folds)}
    for p in sorted((HERE/'runs').glob('*/*/seed_*/*/result.json')):
        result=json.loads(p.read_text());base=p.parent;name=result['dataset']
        pred=np.load(base/'predictions.npz');split=np.load(base/'split.npz')
        classes=len(json.loads((HERE/'data'/name/'manifest.json').read_text())['classes'])
        labels=np.load(HERE/'data'/name/'y.npy')
        assert np.array_equal(split['test'],pred['test_indices'])
        assert np.array_equal(labels[pred['test_indices']],pred['y_true'])
        assert np.array_equal(pred['logits'].argmax(1),pred['y_pred'])
        computed={'accuracy':float(accuracy_score(pred['y_true'],pred['y_pred']))}
        for metric,average in [('f1_macro','macro'),('f1_weighted','weighted')]:
            computed[metric]=float(f1_score(pred['y_true'],pred['y_pred'],labels=list(range(classes)),average=average,zero_division=0))
        for k,v in computed.items():assert abs(v-result['test'][k])<1e-12,(p,k)
        for file,h in result['artifacts'].items():assert sha(base/file)==h,(p,file)
        assert result['epochs_run']<=result['recipe']['max_epochs']
        if result.get('execution_source_sha256'):
            for file,h in result['execution_source_sha256'].items():assert sha(base/'execution_source'/file)==h
        # Audit saved scaler against an independently reconstructed train-only fit.
        if result['method']!='if_convtransformer':
            x=np.load(HERE/'data'/name/'x.npy',mmap_mode='r')[split['train']].copy()
            scaler=np.load(base/'scaler.npz')
            median=np.nanmedian(x.reshape(-1,x.shape[2]),0)
            assert np.allclose(median,scaler['median'],atol=1e-6)
            missing=~np.isfinite(x);x[missing]=np.broadcast_to(median,x.shape)[missing]
            assert np.allclose(x.mean((0,1),dtype=np.float64),scaler['mean'],atol=1e-6)
            assert np.allclose(np.maximum(x.std((0,1),dtype=np.float64),1e-6),scaler['std'],atol=1e-6)
        runs.append({'method':result['method'],'dataset':name,'seed':result['seed'],'fold':result['fold'],
                     'result_sha256':sha(p),'predictions_recomputed':True,'artifact_hashes_checked':True})
    report={'status':'passed','datasets':datasets,'verified_completed_runs':runs,
            'tinyhar_model_sha256':sha(HERE/'source/ISWC22-HAR/models/TinyHAR.py')}
    atomic_json(HERE/'verification.json',report)
    print(json.dumps({'status':'passed','datasets':len(datasets),'subject_splits':sum(x['verified_splits'] for x in datasets.values()),'completed_runs':len(runs)}))


if __name__=='__main__':main()
