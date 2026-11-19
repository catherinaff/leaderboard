"""Reproduce TPN's original self-supervised paper protocol.

TPN's own IMWUT 2019 paper reports F-score and Cohen's kappa for
self-supervised transform-prediction pretraining followed by an activity
classifier. This runner implements that protocol over prepared IMU arrays:

1. train a convolutional encoder to predict signal transformations;
2. reuse the encoder for activity classification;
3. evaluate macro F-score and kappa against Table 2.

When available, it uses LIMU-BERT's prepared dataset arrays only as a
consistent local source of segmented IMU windows. The model/protocol is the TPN
paper protocol, not the LIMU-BERT 1% labeled baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
LIMU_DATASET_ROOT = ROOT / "experiments" / "repos" / "LIMU-BERT-Public" / "dataset"
GAP_THRESHOLD = 0.05


PAPER_REFERENCE = {
    "hhar": {
        "benchmark_id": "hhar",
        "dataset_dir": "hhar",
        "label_index": 2,
        "user_index": 0,
        "paper_metrics": {"f1_weighted": 0.7862, "cohen_kappa": 0.7555},
        "paper_reference_source": "TPN original paper Table 2, Self-Supervised + FT row",
        "paper_dataset_note": "HHAR accelerometer HAR; TPN paper reports F-score and Kappa.",
    },
    "unimib": {
        "benchmark_id": "unimib_shar",
        "dataset_dir": "unimib",
        "label_index": 0,
        "user_index": 1,
        "paper_metrics": {"f1_weighted": 0.8425, "cohen_kappa": 0.8197},
        "paper_reference_source": "TPN original paper Table 2, Self-Supervised + FT row",
        "paper_dataset_note": "UniMiB-SHAR smartphone accelerometer ADL subset with 9 activities; TPN paper reports F-score and Kappa.",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_array(dataset_dir: Path, stem: str) -> Path:
    candidates = []
    candidates.extend(sorted(dataset_dir.glob(f"{stem}_400*.npy")))
    candidates.extend(sorted(dataset_dir.glob(f"{stem}_20_120.npy")))
    candidates.extend(sorted(dataset_dir.glob(f"{stem}*.npy")))
    if not candidates:
        raise FileNotFoundError(f"No {stem} array found under {dataset_dir}")
    return candidates[0]


def normalize_shape(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D IMU array, got shape {x.shape}")
    # Most prepared HAR arrays are (N, T, C). Convert (N, C, T) when obvious.
    if x.shape[1] <= 12 and x.shape[2] > x.shape[1]:
        x = np.transpose(x, (0, 2, 1))
    return x.astype(np.float32, copy=False)


def resample_windows(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[1] == target_len:
        return x
    old_t = np.linspace(0.0, 1.0, x.shape[1], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((x.shape[0], target_len, x.shape[2]), dtype=np.float32)
    for i in range(x.shape[0]):
        for c in range(x.shape[2]):
            out[i, :, c] = np.interp(new_t, old_t, x[i, :, c])
    return out


def load_dataset(name: str, seq_len: int, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    if name == "unimib":
        maybe_mat = load_unimib_mat(seq_len, channels)
        if maybe_mat is not None:
            return maybe_mat

    ref = PAPER_REFERENCE[name]
    dataset_dir = LIMU_DATASET_ROOT / ref["dataset_dir"]
    data_path = find_array(dataset_dir, "data")
    label_path = find_array(dataset_dir, "label")
    x = normalize_shape(np.load(data_path))
    y_raw = np.load(label_path)
    y = collapse_labels(y_raw, int(ref["label_index"]))
    groups = collapse_labels(y_raw, int(ref["user_index"])) if "user_index" in ref else None
    _, y = np.unique(y.astype(np.int64), return_inverse=True)
    if channels and x.shape[2] >= channels:
        x = x[:, :, :channels]
    x = resample_windows(x, seq_len)
    meta = {
        "data_path": str(data_path),
        "label_path": str(label_path),
        "original_shape": list(np.load(data_path, mmap_mode="r").shape),
        "used_shape": list(x.shape),
        "label_shape": list(y_raw.shape),
        "class_count": int(np.unique(y).size),
    }
    return x, y.astype(np.int64), groups, meta


def load_unimib_mat(seq_len: int, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict] | None:
    maybe_npy = load_unimib_npy(seq_len, channels)
    if maybe_npy is not None:
        return maybe_npy

    roots = [
        ROOT / "experiments" / "datasets" / "UniMiB-SHAR" / "data",
        ROOT / "experiments" / "datasets" / "unimib_shar" / "UniMiB-SHAR" / "data",
        ROOT / "experiments" / "datasets" / "UniMiB-SHAR" / "UniMiB-SHAR" / "data",
    ]
    data_dir = next((p for p in roots if (p / "adl_data.mat").exists()), None)
    if data_dir is None:
        return None
    from scipy import io

    parts = []
    labels = []
    groups = []
    raw_shapes = {}
    data_path = data_dir / "adl_data.mat"
    label_path = data_dir / "adl_labels.mat"
    if not data_path.exists() or not label_path.exists():
        return None
    mat_data = io.loadmat(data_path)["adl_data"]
    mat_labels = io.loadmat(label_path)["adl_labels"]
    raw_shapes[data_path.name] = list(mat_data.shape)
    raw_shapes[label_path.name] = list(mat_labels.shape)
    x_part = unimib_flat_to_window(mat_data).astype(np.float32)
    y_part = mat_labels[:, 0].astype(np.int64) - 1
    group_part = mat_labels[:, 1].astype(np.int64) if mat_labels.shape[1] > 1 else np.arange(len(y_part))
    parts.append(x_part)
    labels.append(y_part)
    groups.append(group_part)
    if not parts:
        return None
    x = np.concatenate(parts, axis=0)
    y = np.concatenate(labels, axis=0)
    group = np.concatenate(groups, axis=0) if groups else None
    _, y = np.unique(y, return_inverse=True)
    if channels and x.shape[2] >= channels:
        x = x[:, :, :channels]
    x = resample_windows(x, seq_len)
    meta = {
        "data_path": str(data_dir),
        "raw_shapes": raw_shapes,
        "used_shape": list(x.shape),
        "class_count": int(np.unique(y).size),
        "loader": "UniMiB-SHAR .mat ADL subset; 453 flat samples interpreted as axis-concatenated 151-sample x/y/z blocks.",
        "paper_subset": "ADL only, 9 activities, matching the TPN paper dataset table.",
    }
    return x, y.astype(np.int64), group, meta


def unimib_flat_to_window(data: np.ndarray) -> np.ndarray:
    """Convert UniMiB 453-sample rows to (N, 151, 3) as axis blocks.

    The Kaggle and MATLAB exports store each row as 151 samples from one axis,
    then 151 from the next, then 151 from the third. This is equivalent to
    Fortran-order reshape, but explicit splitting makes the protocol visible.
    """
    if data.ndim != 2 or data.shape[1] != 453:
        raise ValueError(f"Expected flattened UniMiB shape (N, 453), got {data.shape}")
    return np.stack([data[:, 0:151], data[:, 151:302], data[:, 302:453]], axis=2)


def load_unimib_npy(seq_len: int, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict] | None:
    roots = [
        ROOT / "experiments" / "datasets" / "UniMiB-SHAR",
        ROOT / "experiments" / "datasets" / "unimib_shar",
        ROOT / "experiments" / "datasets" / "unimib",
    ]
    root = next((p for p in roots if (p / "adl_data.npy").exists()), None)
    if root is None:
        return None
    data_path = root / "adl_data.npy"
    label_path = root / "adl_labels.npy"
    data = np.load(data_path)
    labels = np.load(label_path)
    if data.ndim == 2 and data.shape[1] == 453:
        x = unimib_flat_to_window(data).astype(np.float32)
    elif data.ndim == 3:
        x = normalize_shape(data)
    else:
        raise ValueError(f"Unsupported UniMiB acc_data shape {data.shape}")
    y = labels[:, 0].astype(np.int64) - 1
    group = labels[:, 1].astype(np.int64) if labels.ndim == 2 and labels.shape[1] > 1 else None
    _, y = np.unique(y, return_inverse=True)
    if channels and x.shape[2] >= channels:
        x = x[:, :, :channels]
    x = resample_windows(x, seq_len)
    meta = {
        "data_path": str(data_path),
        "label_path": str(label_path),
        "original_shape": list(data.shape),
        "label_shape": list(labels.shape),
        "used_shape": list(x.shape),
        "class_count": int(np.unique(y).size),
        "loader": "UniMiB-SHAR Kaggle .npy ADL subset; labels interpreted as activity, subject, trial.",
        "npy_flatten_order": "axis-concatenated 151-sample x/y/z blocks; equivalent to reshape(order='F') for (151, 3).",
        "paper_subset": "ADL only, 9 activities, matching the TPN paper dataset table.",
    }
    return x, y.astype(np.int64), group, meta


def collapse_labels(y_raw: np.ndarray, label_index: int) -> np.ndarray:
    if y_raw.ndim == 1:
        return y_raw
    if y_raw.ndim == 2:
        selected = y_raw[:, label_index] if y_raw.shape[1] > label_index else y_raw[:, 0]
    elif y_raw.ndim == 3:
        selected = y_raw[:, :, label_index] if y_raw.shape[2] > label_index else y_raw[:, :, 0]
    else:
        raise ValueError(f"Unsupported label shape {y_raw.shape}")
    if selected.ndim == 1:
        return selected
    collapsed = []
    for row in selected:
        vals, counts = np.unique(row.astype(np.int64), return_counts=True)
        collapsed.append(vals[np.argmax(counts)])
    return np.asarray(collapsed, dtype=np.int64)


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    split_mode: str,
    test_user_fraction: float,
):
    if groups is not None and split_mode == "user_holdout":
        rng = np.random.default_rng(seed)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        test_count = max(1, int(math.ceil(test_user_fraction * len(unique_groups))))
        test_groups = set(unique_groups[:test_count].tolist())
        test_idx = np.flatnonzero(np.isin(groups, list(test_groups)))
        remaining_idx = np.flatnonzero(~np.isin(groups, list(test_groups)))
        train_idx, val_idx = train_test_split(
            remaining_idx,
            test_size=0.2,
            random_state=seed,
            stratify=y[remaining_idx],
        )
        mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
        std = x[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        x = (x - mean) / std
        return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,
        random_state=seed,
        stratify=y[train_idx],
    )
    mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    x = (x - mean) / std
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(np.transpose(x, (0, 2, 1))).float()
    yt = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle, drop_last=False)


class TPNEncoder(nn.Module):
    def __init__(self, channels: int, feature_layer: int = 3) -> None:
        super().__init__()
        if feature_layer not in (1, 2, 3):
            raise ValueError(f"feature_layer must be 1, 2, or 3, got {feature_layer}")
        self.feature_layer = feature_layer
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(channels, 32, 24, stride=1, padding=12), nn.ReLU(inplace=True), nn.Dropout(0.1)),
                nn.Sequential(nn.Conv1d(32, 64, 16, stride=1, padding=8), nn.ReLU(inplace=True), nn.Dropout(0.1)),
                nn.Sequential(nn.Conv1d(64, 96, 8, stride=1, padding=4), nn.ReLU(inplace=True), nn.Dropout(0.1)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks[: self.feature_layer]:
            out = block(out)
        return torch.amax(out, dim=2)

    @property
    def feature_dim(self) -> int:
        return [32, 64, 96][self.feature_layer - 1]


class TransformPredictor(nn.Module):
    def __init__(self, encoder: TPNEncoder, feature_dim: int, transform_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)) for _ in range(transform_count)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.encoder(x)
        return [head(z).squeeze(1) for head in self.heads]


class ActivityClassifier(nn.Module):
    def __init__(self, encoder: TPNEncoder, feature_dim: int, class_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 1024), nn.ReLU(inplace=True), nn.Linear(1024, class_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def apply_transform(x: torch.Tensor, task_idx: int) -> torch.Tensor:
    out = x.clone()
    b, c, t = out.shape

    if task_idx == 0:
        out = out + 0.05 * torch.randn_like(out)
    elif task_idx == 1:
        scale = torch.empty((b, c, 1), device=x.device).uniform_(0.8, 1.2)
        out = out * scale
    elif task_idx == 2:
        for i in range(b):
            signs = torch.where(torch.rand(c, device=x.device) < 0.5, -1.0, 1.0).view(c, 1)
            if torch.all(signs > 0):
                signs[torch.randint(0, c, (1,), device=x.device)] = -1.0
            out[i] = out[i] * signs
    elif task_idx == 3:
        out = -out
    elif task_idx == 4:
        out = torch.flip(out, dims=[2])
    elif task_idx == 5 and t >= 8:
        segments = 4
        edges = torch.linspace(0, t, segments + 1, dtype=torch.long, device=x.device)
        for i in range(b):
            pieces = [out[i, :, int(edges[j]) : int(edges[j + 1])] for j in range(segments)]
            perm = torch.randperm(segments, device=x.device).tolist()
            out[i] = torch.cat([pieces[j] for j in perm], dim=1)
    elif task_idx == 6:
        base = torch.linspace(0, 1, t, device=x.device)
        for i in range(b):
            alpha = float(torch.empty((), device=x.device).uniform_(0.8, 1.2))
            warped = torch.pow(base, alpha)
            pos = warped * (t - 1)
            left = torch.floor(pos).long().clamp(max=t - 2)
            frac = (pos - left.float()).view(1, -1)
            out[i] = out[i, :, left] * (1 - frac) + out[i, :, left + 1] * frac
    elif task_idx == 7 and c > 1:
        for i in range(b):
            perm = torch.randperm(c, device=x.device)
            out[i] = out[i, perm]
    return out


def train_pretext(
    model: TransformPredictor,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    repeats: int,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n = 0
        for _ in range(max(1, repeats)):
            for xb, _ in loader:
                xb = xb.to(device)
                loss = 0.0
                for task_idx in range(len(model.heads)):
                    transformed = apply_transform(xb, task_idx)
                    task_x = torch.cat([xb, transformed], dim=0)
                    labels = torch.cat(
                        [torch.zeros(xb.size(0), device=device), torch.ones(xb.size(0), device=device)],
                        dim=0,
                    )
                    logits = model(task_x)[task_idx]
                    loss = loss + criterion(logits, labels)
                loss = loss / len(model.heads)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"pretrain epoch {epoch}/{epochs} loss {total / max(1, n):.4f}", flush=True)


def run_classifier(
    model: ActivityClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    weight_decay: float,
) -> tuple[dict[str, float], int]:
    optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["f1_weighted"] > best_f1:
            best_f1 = val_metrics["f1_weighted"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"classifier epoch {epoch}/{epochs} loss {total / max(1, n):.4f} "
                f"val_f1_weighted {val_metrics['f1_weighted']:.4f} val_kappa {val_metrics['cohen_kappa']:.4f}",
                flush=True,
            )
        if stale >= patience:
            print(f"early stopping at epoch {epoch}, best epoch {best_epoch}", flush=True)
            break
    if best_state:
        model.load_state_dict(best_state)
    return evaluate(model, test_loader, device), best_epoch


def freeze_encoder_for_ft(model: ActivityClassifier) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    selected = model.encoder.blocks[model.encoder.feature_layer - 1]
    for param in selected.parameters():
        param.requires_grad = True


def evaluate(model: ActivityClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            ys.append(yb.cpu().numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def calibration_status(metrics: dict[str, float], paper: dict[str, float]) -> tuple[str, str, dict[str, float]]:
    gaps = {k: float(metrics[k]) - float(v) for k, v in paper.items()}
    max_gap = max(abs(v) for v in gaps.values())
    return ("within_expected_range", "passed", gaps) if max_gap <= GAP_THRESHOLD else ("large_gap_check_protocol", "failed", gaps)


@dataclass
class Args:
    datasets: list[str]
    seq_len: int
    channels: int
    seed: int
    batch_size: int
    pretrain_epochs: int
    finetune_epochs: int
    lr: float
    patience: int
    weight_decay: float
    split_mode: str
    test_user_fraction: float
    pretrain_repeats: int
    feature_layer: int
    gpu: int | None
    run_tag: str


def run_dataset(name: str, args: Args) -> dict:
    ref = PAPER_REFERENCE[name]
    set_seed(args.seed)
    x, y, groups, data_meta = load_dataset(name, args.seq_len, args.channels)
    class_count = int(np.unique(y).size)
    train, val, test = split_data(x, y, groups, args.seed, args.split_mode, args.test_user_fraction)
    train_loader = to_loader(*train, batch_size=args.batch_size, shuffle=True)
    val_loader = to_loader(*val, batch_size=args.batch_size, shuffle=False)
    test_loader = to_loader(*test, batch_size=args.batch_size, shuffle=False)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else "cpu")

    started = time.time()
    pretrain_repeats = args.pretrain_repeats if args.pretrain_repeats > 0 else (5 if name == "unimib" else 1)
    feature_layer = args.feature_layer if args.feature_layer > 0 else (2 if name == "hhar" else 3)
    encoder = TPNEncoder(channels=train[0].shape[2], feature_layer=feature_layer).to(device)
    pretext = TransformPredictor(encoder, feature_dim=encoder.feature_dim, transform_count=8).to(device)
    train_pretext(pretext, train_loader, device, args.pretrain_epochs, args.lr, args.weight_decay, pretrain_repeats)
    classifier = ActivityClassifier(pretext.encoder, feature_dim=pretext.encoder.feature_dim, class_count=class_count).to(device)
    freeze_encoder_for_ft(classifier)
    metrics, best_epoch = run_classifier(
        classifier,
        train_loader,
        val_loader,
        test_loader,
        device,
        args.finetune_epochs,
        args.lr,
        args.patience,
        args.weight_decay,
    )
    elapsed = round(time.time() - started, 3)
    gap_status, cal_status, gaps = calibration_status(metrics, ref["paper_metrics"])
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": metrics["f1_weighted"],
        "primary_metric": "f1_weighted",
        "metrics": metrics,
        "source": "tpn_original_paper_protocol_reimplementation",
        "protocol_status": "paper_protocol",
        "official_code_used": False,
        "paper_reimplementation": True,
        "implementation_scope": "TPN original protocol reimplementation: transform-prediction pretraining plus activity classifier fine-tuning.",
        "pretext_transformations": [
            "jitter/noise",
            "scaling",
            "random per-axis sign rotation",
            "negation",
            "horizontal/time flip",
            "segment permutation",
            "time warping",
            "channel shuffle",
        ],
        "implementation_caveat": "Paper-style reimplementation; exact original TPN source code and transform set are still being audited before leaderboard promotion.",
        "architecture": "TPN paper architecture: Conv1D 32/64/96 with kernels 24/16/8, stride 1, ReLU, dropout 0.1, global max pooling; per-task FC-256 sigmoid heads; activity classifier FC-1024 softmax with the selected final convolution block fine-tuned.",
        "dataset_source": "LIMU-BERT prepared IMU windows when available",
        "dataset_meta": data_meta,
        "split": "user-held-out 70/10/20 over subjects with stratified train/validation" if args.split_mode == "user_holdout" else "stratified random 70/10/20 over windows",
        "split_mode": args.split_mode,
        "test_user_fraction": args.test_user_fraction if args.split_mode == "user_holdout" else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_repeats": pretrain_repeats,
        "feature_layer": f"Conv{feature_layer}",
        "finetune_epochs": args.finetune_epochs,
        "best_finetune_epoch": best_epoch,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "runtime_seconds": elapsed,
        "paper_reference_metric": "f1_weighted",
        "paper_reference_score": ref["paper_metrics"]["f1_weighted"],
        "paper_reference_metrics": ref["paper_metrics"],
        "paper_reference_source": ref["paper_reference_source"],
        "paper_reference_dataset_note": ref["paper_dataset_note"],
        "paper_reference_metric_gaps": gaps,
        "paper_reference_gap_abs": abs(gaps["f1_weighted"]),
        "paper_reference_gap_signed": gaps["f1_weighted"],
        "paper_reference_max_gap_abs": max(abs(v) for v in gaps.values()),
        "paper_reference_gap_threshold": GAP_THRESHOLD,
        "paper_reference_gap_status": gap_status,
        "calibration_status": cal_status,
        "protocol_note": "TPN original paper-style self-supervised transform prediction followed by activity classification; reports F-score and Cohen's kappa.",
    }
    if cal_status != "passed":
        result["review_note"] = "Outside 5-point tolerance versus TPN Table 2; inspect dataset windowing/split and exact original transform/task implementation before promotion."
    out_dir = RUNS_DIR / f"{ref['benchmark_id']}_tpn_original_paper_protocol"
    write_json(out_dir / "results.json", [{"id": "tpn", "results": [result]}])
    print(
        f"{name}: weighted_F1={metrics['f1_weighted']:.4f} paper={ref['paper_metrics']['f1_weighted']:.4f} "
        f"gap={gaps['f1_weighted']:+.4f}; macro_F1={metrics['f1_macro']:.4f}; kappa={metrics['cohen_kappa']:.4f} "
        f"paper={ref['paper_metrics']['cohen_kappa']:.4f} gap={gaps['cohen_kappa']:+.4f}; "
        f"status={cal_status}",
        flush=True,
    )
    return result


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hhar", "unimib"], choices=sorted(PAPER_REFERENCE))
    parser.add_argument("--seq-len", type=int, default=400)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split-mode", choices=["user_holdout", "stratified_random"], default="user_holdout")
    parser.add_argument("--test-user-fraction", type=float, default=0.3)
    parser.add_argument("--pretrain-repeats", type=int, default=0, help="0 uses paper defaults: UniMiB=5, other datasets=1")
    parser.add_argument("--feature-layer", type=int, choices=[0, 1, 2, 3], default=0, help="0 uses paper notes: HHAR=Conv2, other datasets=Conv3")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-tag", default=f"tpn_original_paper_protocol_{int(time.time())}")
    ns = parser.parse_args()
    return Args(**vars(ns))


def main() -> int:
    args = parse_args()
    results = []
    for dataset in args.datasets:
        results.append(run_dataset(dataset, args))
    aggregate = RUNS_DIR / f"{args.run_tag}.json"
    entry = {
        "id": "tpn",
        "method_name": "TPN",
        "year": 2019,
        "method_family": "Self-supervised transform prediction",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_reimplementation_pending_calibration",
        "summary": "Transformation Prediction Network reproduction under the original paper-style self-supervised protocol.",
        "implementation": "Convolutional encoder pretrained on IMU transformations, then fine-tuned with an activity classifier; evaluated by macro F-score and Cohen's kappa.",
        "har_scope": "Self-supervised IMU human activity recognition.",
        "results": results,
        "sources": [
            "https://arxiv.org/abs/1907.11879",
            "https://doi.org/10.1145/3341162.3344692",
        ],
    }
    write_json(aggregate, [entry])
    print(f"Wrote aggregate: {aggregate}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
