"""Author CDNet encoder, trained on canonical native-label HAR splits.

This is a supervised CDNet adapter, without the unpublished LLM/synthetic-data
training workflow. The released uniHAR IMU weights are deliberately not loaded.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType, MethodType
import fcntl
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RELEASE = HERE.parent / 'onehar_20261028'
OFFICIAL = RELEASE / 'official'
RUNTIME = RELEASE / 'runtime'
COMMIT = 'e98bc56bbd575a61042d85825f4de85f07c594db'
LIMU = OFFICIAL / 'saved/pretrain_base_hhar_20_120/hhar.pt'
TEXT_CHECKPOINT = RELEASE / 'artifacts/uniHAR.pth'
TEXT_CHECKPOINT_SHA = '66f2bf6af6497c2a984a4f0b4f1cc9ee5d2549dcf5bf17401638137c0daa51bf'
ACC = {'uci_har': [[6, 7, 8]], 'pamap2': [[0, 1, 2], [9, 10, 11], [18, 19, 20]],
       'usc_had': [[0, 1, 2]],
       'realworld_har': [[6*i, 6*i+1, 6*i+2] for i in range(7)]}


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def author_modules():
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location('onehar_author_test', RUNTIME / 'test.py')
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    # The author denominator sums over the batch as well as sensor positions.
    # Correct that axis for independent per-window predictions, and audit parity
    # against this explicitly patched author forward (not the released scores).
    text = (OFFICIAL / 'HARmodel.py').read_text()
    old = 'weighted_average = weighted_embedded_positions.sum(dim=1, keepdim=True) / weights.sum()'
    assert text.count(old) == 1
    text = text.replace(old, 'weighted_average = weighted_embedded_positions.sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).unsqueeze(-1)')
    har = ModuleType('onehar_batch_independent_author')
    exec(compile(text, str(OFFICIAL / 'HARmodel.py'), 'exec'), har.__dict__)
    from config import PretrainModelConfig
    cfg = PretrainModelConfig.from_json(json.loads((OFFICIAL / 'config/limu_bert.json').read_text())['base_v1'])
    args = SimpleNamespace(model_cfg=cfg)
    return test, har, args


def label_vectors(test, labels, device):
    cache = HERE / 'assets'
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(labels).encode()).hexdigest()
    dest = cache / (key + '.pt')
    with (cache / 'text.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dest.exists():
            assert sha(TEXT_CHECKPOINT) == TEXT_CHECKPOINT_SHA
            encoder = test.LabelEmbeddingEncoder(labels)
            state = torch.load(TEXT_CHECKPOINT, map_location='cpu', weights_only=True)
            # Only the frozen natural-language BERT is used from this file.
            # No CDNet, projection, discriminator or IMU weights are loaded.
            state = {k.removeprefix('label_encoder.bert.'): v for k, v in state.items()
                     if k.startswith('label_encoder.bert.')}
            assert state
            encoder.bert.load_state_dict(state, strict=True)
            encoder.to(device).eval()
            with torch.no_grad():
                vectors = encoder(torch.arange(len(labels), device=device)).cpu()
            assert torch.isfinite(vectors).all()
            torch.save({'vectors': vectors, 'labels': labels,
                        'source_sha256': TEXT_CHECKPOINT_SHA, 'role': 'frozen BERT text weights only'}, dest)
            del encoder, state
        data = torch.load(dest, map_location='cpu', weights_only=True)
        assert data['labels'] == labels and data['source_sha256'] == TEXT_CHECKPOINT_SHA
    return data['vectors'], sha(dest)


def input_acc(x, dataset):
    """Keep every native window/label; adapt only its sensor representation."""
    result = np.stack([x[:, :, ids] for ids in ACC[dataset]], axis=1)
    n, positions, steps, _ = result.shape
    if steps != 120:
        value = torch.from_numpy(result).reshape(n*positions, steps, 3).transpose(1, 2)
        result = F.interpolate(value, size=120, mode='linear', align_corners=False).transpose(1, 2)
        result = result.reshape(n, positions, 120, 3).numpy()
    return np.ascontiguousarray(result, dtype=np.float32)


def load_limu(device, args):
    import models
    model = models.LIMUBertModel4Pretrain(args.model_cfg, output_embed=True)
    model.load_state_dict(torch.load(LIMU, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False).to(device).eval()
    return model


def limu_embed(limu, value):
    # Exact author Preprocess4Normalization(feature_len=6): only the first
    # acceleration triplet is divided by 9.8. The repeated triplet is retained.
    value = value.detach().clone()
    value[:, :, :3] /= 9.8
    return limu(value)


def frozen_pool(core, limu, acc, device, batch=128):
    n, positions = acc.shape[:2]
    result = np.empty((n, 120, 72), np.float32)
    with torch.no_grad():
        for begin in range(0, n, batch):
            raw = torch.from_numpy(acc[begin:begin+batch]).to(device)
            pooled = torch.zeros((len(raw), 120, 72), device=device)
            for j in range(positions):
                value = core.limuIn_layer_norm(raw[:, j])
                value = core.limuIn_layer_norm(value + core.pos_embed)
                pooled.add_(limu_embed(limu, value.repeat(1, 1, 2)), alpha=j+1)
            pooled /= positions*(positions+1)/2
            result[begin:begin+len(raw)] = pooled.cpu().numpy()
            if begin == 0 or begin//batch % 100 == 0:
                print('FROZEN_FEATURES', begin+len(raw), '/', n, flush=True)
    return result


def cached_forward(core, packed, args=None):
    b, t, channels = packed.shape
    positions = (channels-72)//3
    imu = packed[:, :, :3*positions].reshape(b, t, positions, 3).transpose(1, 2)
    embedded = packed[:, :, 3*positions:].unsqueeze(1)
    imu = core.input_layer_norm(imu.reshape(-1, 120, 3)).reshape(b, positions, 120, 3)
    axes = [vit(imu[:, :, :, j]) for j, vit in enumerate([core.vit_x, core.vit_y, core.vit_z])]
    value = torch.cat(axes, dim=1).permute(0, 2, 1)
    value, _ = core.gru1(value)
    value, _ = core.gru2(value)
    value = torch.cat([embedded, value.unsqueeze(1)], dim=1)
    return core.output_layer_norm(value.reshape(-1, 120, 72)).reshape(b, 2, 120, 72)


class CanonicalCDNet(nn.Module):
    def __init__(self, author_encoder, vectors):
        super().__init__()
        self.imu_encoder = author_encoder
        self.register_buffer('label_vectors', F.normalize(vectors.float(), dim=-1))

    def forward(self, value):
        features = F.normalize(self.imu_encoder(value), dim=-1)
        return (features @ self.label_vectors.T) / .07

    def training_loss(self, value, labels):
        return F.cross_entropy(self(value), labels)


def parity_check(encoder, har, core, limu, acc, device):
    """Cached and uncached author paths, and batch independence, must agree."""
    probe = torch.from_numpy(acc[:3]).to(device)
    def callback(args, save=False, output_embed=True, sensor_data=None):
        return None, limu_embed(limu, sensor_data)
    har.generate_embedding_or_output = callback
    encoder.eval()
    original = core.forward
    with torch.no_grad():
        expected = encoder(probe)
        pooled = frozen_pool(core, limu, acc[:3], device)
        packed = np.concatenate([acc[:3].transpose(0, 2, 1, 3).reshape(3, 120, -1), pooled], axis=-1)
        value = torch.from_numpy(packed).to(device)
        core.forward = MethodType(cached_forward, core)
        actual = encoder(value)
        split = torch.cat([encoder(value[i:i+1]) for i in range(3)])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual, split, rtol=2e-5, atol=2e-5)
    return {'cached_vs_patched_author_max_abs': float((actual-expected).abs().max()),
            'batch_independence_max_abs': float((actual-split).abs().max()),
            'author_pool_axis_fix': True, 'uncached_source_sha256': sha(OFFICIAL/'HARmodel.py')}


def build(x, train, validation, out, a, device):
    labels = json.loads((HERE/'data'/a.dataset/'manifest.json').read_text())['classes']
    test, har, args = author_modules()
    # Building/caching BERT changes RNG consumption only on the first run. Restore
    # RNG so every fold's model initialization depends only on its stated seed.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    vectors, vector_sha = label_vectors(test, labels, device)
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng, device)
    core = har.HARModel(args)
    positions = len(ACC[a.dataset])
    if positions > 5:
        for vit in [core.vit_x, core.vit_y, core.vit_z]:
            vit.pos_embed = nn.Parameter(torch.empty(1, positions+1, 120))
            nn.init.trunc_normal_(vit.pos_embed, std=.02)
    core.limuIn_layer_norm.requires_grad_(False)
    core.pos_embed.requires_grad_(False)
    encoder = test.HARModelWrapper(core, args).to(device)
    limu = load_limu(device, args)
    acc = input_acc(x, a.dataset)
    used = np.arange(len(x))
    if a.smoke:
        with np.load(out/'split.npz') as split:
            used = np.unique(np.concatenate([split[k] for k in ['train', 'validation', 'test']]))
    proof = parity_check(encoder, har, core, limu, acc[used[:3]], device)
    # core.forward was replaced only after uncached/cached parity passed.
    frozen = frozen_pool(core, limu, acc[used], device)
    packed = np.zeros((len(x), 120, 3*positions+72), np.float32)
    packed[used] = np.concatenate([acc[used].transpose(0, 2, 1, 3).reshape(len(used), 120, -1), frozen], axis=-1)
    net = CanonicalCDNet(encoder, vectors).to(device)
    del limu, frozen, acc
    torch.cuda.empty_cache()
    recipe = {
        'optimizer': 'Adam', 'lr': .001, 'weight_decay': 0., 'batch_size': 256,
        'max_epochs': 80, 'loss': 'cross_entropy on normalized IMU/text cosine logits, temperature=.07',
        'architecture': 'author CDNet: frozen HHAR LIMU-BERT + three axis ViTs + BiGRU18/72 + Conv2d projection768 + frozen BERT class text',
        'implementation_type': 'author_architecture_with_disclosed_supervised_adapter',
        'official_repo': 'https://github.com/ZJU-wqx-2025/uniHAR', 'official_commit': COMMIT,
        'variant_label': 'CDNet supervised adapter; no LLM synthesis, DQA/DPS/PDS or cross-dataset joint training',
        'model_sha256': sha(__file__), 'input': 'B,T,C', '_input_array': np.ascontiguousarray(packed),
        'channel_indices': ACC[a.dataset],
        'sensor_adapter': 'All canonical windows and native class labels retained; author acceleration-only sensor input; full window linearly resampled to 120 steps; no test-fitted normalization. UCI total acceleration indices6:9; PAMAP all3 positions; RealWorld all7 positions; USC hip.',
        'compatibility': 'Batch-independent sensor-weight normalization corrects author batch-axis denominator. RealWorld positional capacity extended from5 to7 sensors. Detached LIMU branch cached exactly; parity audited for each fold.',
        'calibration': 'Unified-dataset supervised CDNet variant, not a complete reproduction of the paper joint-training/LLM augmentation protocol.',
        'initialization': 'CDNet, ViTs, BiGRUs and IMU projection newly initialized; released uniHAR IMU/projection weights NOT loaded. Author HHAR-only LIMU encoder and frozen BERT text weights retained.',
        'text': {'labels': labels, 'vectors_sha256': vector_sha, 'checkpoint_sha256': TEXT_CHECKPOINT_SHA, 'loaded_prefix': 'label_encoder.bert only'},
        'pretraining': {'LIMU_source': 'author saved/pretrain_base_hhar_20_120/hhar.pt', 'sha256': sha(LIMU)},
        'source_sha256': {str(p.relative_to(RELEASE)): sha(p) for p in [OFFICIAL/'HARmodel.py', OFFICIAL/'VITmodel.py', OFFICIAL/'models.py', OFFICIAL/'utils.py', RUNTIME/'test.py']},
        'cache_audit': proof, 'paid_api_calls': 0,
    }
    return net, recipe
