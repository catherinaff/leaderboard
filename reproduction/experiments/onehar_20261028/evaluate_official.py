"""Reproduce the released uniHAR checkpoint, preserving its test-batch semantics."""
from pathlib import Path
import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
COMMIT = 'e98bc56bbd575a61042d85825f4de85f07c594db'
CHECKPOINT_SHA = '66f2bf6af6497c2a984a4f0b4f1cc9ee5d2549dcf5bf17401638137c0daa51bf'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def setup_runtime():
    official, runtime = ROOT / 'official', ROOT / 'runtime'
    runtime.mkdir(exist_ok=True)
    for item in official.iterdir():
        if item.suffix == '.py':
            shutil.copy2(item, runtime / item.name)
    for name in ['config', 'dataset', 'saved']:
        shutil.copytree(official / name, runtime / name, dirs_exist_ok=True)
    text = (official / 'test.py').read_text()
    replacements = {
        "'/root/autodl-fs/HAR-Genie/dataset/data_config.json'": repr(str(runtime / 'dataset/data_config.json')),
        "'/root/autodl-fs/HAR-Genie/{mode}/{dataset_name}/data_50_120.npy'": repr(str(official / '{mode}/{dataset_name}/data_50_120.npy')),
        "'/root/autodl-fs/HAR-Genie/{mode}/{dataset_name}/label_50_120.npy'": repr(str(official / '{mode}/{dataset_name}/label_50_120.npy')),
        "'dataset': [DATASETS[i] for i in dataset_labels]\n": "'dataset': [DATASETS[i] for i in dataset_labels],\n",
        'from transformers import BertModel, BertTokenizer': 'from transformers import BertModel, BertTokenizer, BertConfig',
        "BertTokenizer.from_pretrained('bert-base-uncased')": "BertTokenizer(vocab_file=" + repr(str(ROOT / 'artifacts/vocab.txt')) + ", do_lower_case=True)",
        "BertModel.from_pretrained('bert-base-uncased')": "BertModel(BertConfig(attn_implementation='eager'))",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError('Upstream code changed: ' + old)
        text = text.replace(old, new)
    (runtime / 'test.py').write_text(text)
    diff = ''.join(difflib.unified_diff((official / 'test.py').read_text().splitlines(True), text.splitlines(True), fromfile='official/test.py', tofile='runtime/test.py'))
    (ROOT / 'runtime_changes.diff').write_text(diff)
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results_official')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--suite', choices=['official', 'adapted'], default='official')
    parser.add_argument('--datasets', help='Comma-separated adapted dataset IDs')
    parser.add_argument('--uci-acc', choices=['total', 'body'], default='total')
    options = parser.parse_args()
    output = ROOT / options.output
    output.mkdir(exist_ok=True)
    assert digest(ROOT / 'artifacts/uniHAR.pth') == CHECKPOINT_SHA
    runtime = setup_runtime()
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from config import PretrainModelConfig
    from types import SimpleNamespace
    torch.set_num_threads(4)
    torch.manual_seed(3431)
    np.random.seed(3431)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('official_test_runtime', runtime / 'test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        model_cfg=PretrainModelConfig.from_json(json.loads((runtime / 'config/limu_bert.json').read_text())['base_v1']),
        train_cfg=str(runtime / 'config/pretrain.json'), gpu=None,
        pretrain_model=str(runtime / 'saved/pretrain_base_hhar_20_120/hhar'),
        save_path=str(output / 'unused'), model_file='hhar', dataset='HHAR', dataset_version='50_120')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    module.device = device
    checkpoint = torch.load(ROOT / 'artifacts/uniHAR.pth', map_location='cpu', weights_only=True)
    model = module.initialize_model(args)
    incompatible = model.load_state_dict(checkpoint, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.to(device).eval()
    load_audit = {'strict_load': True, 'tensor_count': len(checkpoint), 'parameter_count': sum(x.numel() for x in model.parameters()), 'checkpoint_sha256': CHECKPOINT_SHA}
    (output / 'checkpoint_audit.json').write_text(json.dumps(load_audit, indent=2))
    print('CHECKPOINT', json.dumps(load_audit), flush=True)
    del checkpoint

    # The original constructs and fully reloads LIMU-BERT on every forward call.
    # Cache that frozen encoder, while keeping its NumPy pipeline and 128 sub-batches.
    import HARmodel
    import embedding
    original_embedding = embedding.generate_embedding_or_output
    _, _, limu, _, train_cfg = embedding.fetch_setup(args, True, sensor_data=torch.zeros(1, 120, 6))
    limu.load_state_dict(torch.load(args.pretrain_model + '.pt', map_location='cpu', weights_only=True), strict=True)
    limu.to(device).eval()

    def cached_embedding(args, save=False, output_embed=True, sensor_data=None):
        data = sensor_data.detach().cpu().numpy()
        dataset = embedding.IMUDataset(data, pipeline=[embedding.Preprocess4Normalization(6)])
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=train_cfg.batch_size)
        result = []
        with torch.no_grad():
            for batch in loader:
                result.append(limu(batch.to(device)))
        return data, torch.cat(result).cpu().numpy()

    with torch.no_grad():
        probe = torch.randn(19, 120, 6, device=device)
        _, reference = original_embedding(args, sensor_data=probe)
        _, cached = cached_embedding(args, sensor_data=probe)
        np.testing.assert_allclose(cached, reference, rtol=0, atol=0)
    cache_audit = {'limu_cached_vs_original_max_abs_diff': float(np.max(np.abs(cached-reference))), 'frozen_label_embeddings_cached': not options.reference}
    print('CACHE_PARITY', json.dumps(cache_audit), flush=True)
    if not options.reference:
        HARmodel.generate_embedding_or_output = cached_embedding
        with torch.no_grad():
            label_embeddings = model.label_encoder(torch.arange(len(module.LABEL_MAP), device=device))
        original_label_forward = model.label_encoder.forward
        model.label_encoder.forward = lambda labels: label_embeddings[labels]

    if options.suite == 'adapted':
        from evaluate_adapted import run
        run(model, args, module, output, original_label_forward, load_audit, cache_audit, options)
        return

    dataset = module.MultiDatasetLoader(module.DATASETS, module.LABEL_MAP, mode='test')
    sampler = module.SensorCountBatchSampler(dataset, batch_size=options.batch_size, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    batch_indices = list(sampler)
    manifest = []
    for name in module.DATASETS:
        p = ROOT / 'official/test' / name
        x, y = np.load(p / 'data_50_120.npy', mmap_mode='r'), np.load(p / 'label_50_120.npy', mmap_mode='r')
        manifest.append({'dataset':name,'data_shape':list(x.shape),'label_shape':list(y.shape),'data_dtype':str(x.dtype),'labels_at_first_timestep':np.unique(y[:,0,0]).tolist(),'data_sha256':digest(p/'data_50_120.npy'),'label_sha256':digest(p/'label_50_120.npy')})
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DATASETS', json.dumps(manifest), flush=True)
    records, logits_all = [], []
    start = time.time()
    with torch.no_grad():
        for batch_id, (imu, labels, domains, sensor_count) in enumerate(loader):
            if options.max_batches is not None and batch_id >= options.max_batches:
                break
            imu, labels = imu.to(device), labels.to(device)
            features = torch.nn.functional.normalize(model.imu_encoder(imu, args), dim=-1)
            vectors = model.label_encoder(torch.arange(len(module.LABEL_MAP), device=device))
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            logits = (features @ vectors.T).cpu().numpy()
            predictions = logits.argmax(axis=1)
            for i, (label, domain, pred) in enumerate(zip(labels.cpu().numpy(), domains.numpy(), predictions)):
                records.append({'sample_id':int(batch_indices[batch_id][i]),'dataset':module.DATASETS[int(domain)],'true_id':int(label),'pred_id':int(pred),'true_label':module.LABEL_MAP[int(label)],'pred_label':module.LABEL_MAP[int(pred)],'batch_id':batch_id})
            logits_all.append(logits)
            if batch_id % 10 == 0:
                print('PROGRESS', batch_id+1, '/', len(loader), 'samples', len(records), 'seconds', round(time.time()-start,1), flush=True)
    with (output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    np.savez_compressed(output / 'logits.npz', logits=np.concatenate(logits_all))
    results = []
    for name in module.DATASETS:
        rows = [r for r in records if r['dataset'] == name]
        if not rows:
            continue
        true, pred = [r['true_id'] for r in rows], [r['pred_id'] for r in rows]
        present = sorted(set(true))
        results.append({'dataset':name,'samples':len(rows),'classes':len(present),'accuracy':accuracy_score(true,pred),'f1_macro':f1_score(true,pred,average='macro',zero_division=0),'f1_macro_dataset_labels':f1_score(true,pred,labels=present,average='macro',zero_division=0),'f1_weighted':f1_score(true,pred,average='weighted',zero_division=0),'classification_report':classification_report(true,pred,labels=list(range(9)),target_names=module.LABEL_MAP,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(true,pred,labels=list(range(9))).tolist()})
    report = {'repository':'https://github.com/ZJU-wqx-2025/uniHAR','commit':COMMIT,'device':str(device),'torch':torch.__version__,'batch_size':options.batch_size,'full_evaluation':options.max_batches is None,'protocol':'author-released five test sets; fixed released checkpoint; pooled test-position scalar normalization; global nine-class candidates; original sensor-count grouping and batch-size-dependent pooling retained','training_provenance':'Released checkpoint training split is not documented sufficiently to claim LODO or leakage-free canonical evaluation.','not_canonical_leaderboard_protocol':True,'cache_audit':cache_audit,'checkpoint_audit':load_audit,'elapsed_seconds':time.time()-start,'total_samples':len(records),'results':results}
    (output / 'results.json').write_text(json.dumps(report, indent=2))
    print('RESULTS',json.dumps([{k:v for k,v in r.items() if k not in ['confusion_matrix','classification_report']} for r in results]),flush=True)


if __name__ == '__main__':
    main()
