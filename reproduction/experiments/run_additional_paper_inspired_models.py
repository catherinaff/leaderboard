"""Run additional paper-inspired HAR model families on canonical windows.

These are not official author-code reproductions. They are compact PyTorch
baselines that follow the broad architecture family of registry methods whose
released code is blocked by legacy runtimes, missing entrypoints, or dataset
adapter gaps. Result rows therefore carry `official_code_used: false`.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from har_utils import DATASETS_DIR, RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, write_json


UCI_SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

METHODS = {
    "attend_and_discriminate": {
        "name": "Attend And Discriminate",
        "family": "Attention + CNN/GRU",
        "variant": "Local CNN + squeeze/excitation channel attention + GRU attention classifier; no center loss or mixup.",
        "sources": ["https://arxiv.org/abs/2007.07172"],
    },
    "dense_labeling": {
        "name": "Dense Labeling",
        "family": "Fully Convolutional Dense Prediction",
        "variant": "Local fully convolutional temporal dense-labeling proxy with per-sample logits averaged to a window label.",
        "sources": ["https://arxiv.org/abs/2007.07172"],
    },
    "hart": {
        "name": "HART",
        "family": "Lightweight Transformer",
        "variant": "Local lightweight temporal transformer with convolutional tokenization; official TensorFlow HART code was not adapted.",
        "sources": ["https://arxiv.org/abs/2209.11750"],
    },
    "mobilehart": {
        "name": "MobileHART",
        "family": "CNN + Lightweight Transformer",
        "variant": "Local multi-scale convolution plus lightweight transformer baseline inspired by MobileHART; official code was not adapted.",
        "sources": ["https://arxiv.org/abs/2209.11750"],
    },
    "limu_bert": {
        "name": "LIMU-BERT",
        "family": "Self-supervised BERT for IMU",
        "variant": "Local supervised transformer classifier with a CLS token; no masked IMU pretraining or official checkpoint.",
        "sources": ["https://tanrui.github.io/pub/LIMU_BERT.pdf", "https://github.com/dapowan/LIMU-BERT-Public"],
    },
    "if_convtransformer": {
        "name": "IF-ConvTransformer",
        "family": "IMU Fusion + ConvTransformer",
        "variant": "Local convolutional IMU fusion stem followed by a transformer encoder; official repository checkout remains incomplete.",
        "sources": ["https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022"],
    },
    "deepsense": {
        "name": "DeepSense",
        "family": "CNN + RNN framework",
        "variant": "Local CNN + GRU mobile-sensing classifier inspired by DeepSense; no TensorFlow 1.1 official pipeline.",
        "sources": ["https://arxiv.org/abs/1611.01942", "https://github.com/yscacaca/DeepSense"],
    },
    "tinyhar": {
        "name": "TinyHAR",
        "family": "Lightweight CNN/RNN for edge HAR",
        "variant": "Local compact depthwise-separable CNN + GRU baseline inspired by TinyHAR; no official usage script.",
        "sources": ["https://publikationen.bibliothek.kit.edu/1000150216/149250239", "https://github.com/teco-kit/ISWC22-HAR"],
    },
}


class PositionalMixin:
    def add_pos(self, x: torch.Tensor, pos: nn.Parameter) -> torch.Tensor:
        return x + pos[:, : x.shape[1], :]


class DenseLabelingNet(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=-1)


class AttendDiscriminateNet(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int) -> None:
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(4, channels // 2), 1),
            nn.ReLU(),
            nn.Conv1d(max(4, channels // 2), channels, 1),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.fc = nn.Linear(hidden * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_gate(x)
        out, _ = self.gru(self.conv(x).transpose(1, 2))
        weights = torch.softmax(self.attn(out), dim=1)
        return self.fc((weights * out).sum(dim=1))


class HARTLite(nn.Module, PositionalMixin):
    def __init__(self, channels: int, classes: int, hidden: int, layers: int, heads: int, max_len: int) -> None:
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x).transpose(1, 2)
        tokens = self.add_pos(tokens, self.pos)
        return self.fc(self.encoder(tokens).mean(dim=1))


class MobileHARTLite(nn.Module, PositionalMixin):
    def __init__(self, channels: int, classes: int, hidden: int, layers: int, heads: int, max_len: int) -> None:
        super().__init__()
        branch = max(8, hidden // 3)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(channels, branch, 3, padding=1),
                nn.Conv1d(channels, branch, 5, padding=2),
                nn.Conv1d(channels, branch, 7, padding=3),
            ]
        )
        self.proj = nn.Sequential(nn.GELU(), nn.Conv1d(branch * 3, hidden, 1), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        tokens = self.proj(x).transpose(1, 2)
        tokens = self.add_pos(tokens, self.pos)
        return self.fc(self.encoder(tokens).mean(dim=1))


class LIMUBERTLite(nn.Module, PositionalMixin):
    def __init__(self, channels: int, classes: int, hidden: int, layers: int, heads: int, max_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(channels, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, hidden))
        enc_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x.transpose(1, 2))
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.add_pos(tokens, self.pos)
        return self.fc(self.encoder(tokens)[:, 0])


class IFConvTransformerLite(nn.Module, PositionalMixin):
    def __init__(self, channels: int, classes: int, hidden: int, layers: int, heads: int, max_len: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.fusion(x).transpose(1, 2)
        tokens = self.add_pos(tokens, self.pos)
        return self.fc(self.encoder(tokens).mean(dim=1))


class DeepSenseLite(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.conv(x).transpose(1, 2))
        return self.fc(out[:, -1])


class DepthwiseSeparable(nn.Module):
    def __init__(self, channels: int, out_channels: int, kernel: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels),
            nn.Conv1d(channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyHARLite(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int) -> None:
        super().__init__()
        small = max(16, hidden // 2)
        self.conv = nn.Sequential(
            DepthwiseSeparable(channels, small, 5),
            DepthwiseSeparable(small, hidden, 5),
            nn.Dropout(0.1),
        )
        self.gru = nn.GRU(hidden, small, batch_first=True)
        self.fc = nn.Linear(small, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.conv(x).transpose(1, 2))
        return self.fc(out[:, -1])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(method: str, channels: int, classes: int, window_len: int, args) -> nn.Module:
    hidden = args.hidden_dim
    max_len = max(window_len + 1, args.max_len)
    heads = args.heads
    if hidden % heads != 0:
        raise SystemExit("--hidden-dim must be divisible by --heads")
    if method == "attend_and_discriminate":
        return AttendDiscriminateNet(channels, classes, hidden)
    if method == "dense_labeling":
        return DenseLabelingNet(channels, classes, hidden)
    if method == "hart":
        return HARTLite(channels, classes, hidden, args.transformer_layers, heads, max_len)
    if method == "mobilehart":
        return MobileHARTLite(channels, classes, hidden, args.transformer_layers, heads, max_len)
    if method == "limu_bert":
        return LIMUBERTLite(channels, classes, hidden, args.transformer_layers, heads, max_len)
    if method == "if_convtransformer":
        return IFConvTransformerLite(channels, classes, hidden, args.transformer_layers, heads, max_len)
    if method == "deepsense":
        return DeepSenseLite(channels, classes, hidden)
    if method == "tinyhar":
        return TinyHARLite(channels, classes, hidden)
    raise ValueError(method)


def load_uci_har() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "uci_har" / "UCI HAR Dataset"
    if not (data_dir / "train" / "Inertial Signals").is_dir():
        raise SystemExit("UCI-HAR data missing. Run experiments/download_uci_har.py first.")

    xs = []
    ys = []
    subjects = []
    split_masks = []
    for split, is_train in (("train", True), ("test", False)):
        sig_dir = data_dir / split / "Inertial Signals"
        arrays = [np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32) for name in UCI_SIGNALS]
        x_split = np.stack(arrays, axis=2)
        y_split = np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=np.int64) - 1
        s_split = np.loadtxt(data_dir / split / f"subject_{split}.txt", dtype=np.int64)
        xs.append(x_split)
        ys.append(y_split)
        subjects.append(s_split)
        split_masks.append(np.full(len(y_split), is_train, dtype=bool))
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.int64),
        np.concatenate(subjects).astype(np.int64),
        {"train_mask": np.concatenate(split_masks), "split": "official_fixed"},
    )


def load_dataset(benchmark_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if benchmark_id == "uci_har":
        return load_uci_har()
    return load_windows(benchmark_id)


def fixed_subject_split(subjects: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    if "train_mask" in meta:
        train_idx = np.asarray(meta["train_mask"], dtype=bool)
        return train_idx, ~train_idx
    train_subjects = set(int(s) for s in meta.get("train_subjects", []))
    test_subjects = set(int(s) for s in meta.get("test_subjects", []))
    if not train_subjects or not test_subjects:
        raise SystemExit("Fixed split dataset requires train_subjects/test_subjects or train_mask metadata.")
    return np.asarray([int(s) in train_subjects for s in subjects]), np.asarray([int(s) in test_subjects for s in subjects])


def normalize_split(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def train_eval(method: str, x_train, y_train, x_test, y_test, args, device: torch.device) -> tuple[dict, float]:
    x_train, x_test = normalize_split(x_train, x_test)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    classes = int(max(y_train.max(), y_test.max()) + 1)
    model = make_model(method, x_train.shape[1], classes, x_train.shape[2], args).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{METHODS[method]['name']} epoch={epoch} loss={np.mean(losses):.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for start_idx in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start_idx : start_idx + args.batch_size]).to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return metric_bundle(y_test, np.concatenate(preds)), time.time() - start


def evaluate_dataset(method: str, benchmark_id: str, args, device: torch.device) -> tuple[dict, list[dict], float]:
    x, y, subjects, meta = load_dataset(benchmark_id)
    bench = load_benchmark(benchmark_id)
    protocol_type = bench.get("protocol", {}).get("type", "")
    if "Leave-One-Subject-Out" not in protocol_type:
        train_idx, test_idx = fixed_subject_split(subjects, meta)
        metrics, runtime = train_eval(method, x[train_idx], y[train_idx], x[test_idx], y[test_idx], args, device)
        return metrics, [
            {
                "split": "fixed_subject_split",
                "train_windows": int(train_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
            }
        ], runtime

    fold_rows = []
    metric_rows = []
    total_runtime = 0.0
    for subject in sorted(set(int(s) for s in subjects)):
        test_idx = subjects == subject
        train_idx = ~test_idx
        metrics, runtime = train_eval(method, x[train_idx], y[train_idx], x[test_idx], y[test_idx], args, device)
        metric_rows.append(metrics)
        total_runtime += runtime
        fold_rows.append(
            {
                "test_subject": int(subject),
                "train_windows": int(train_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
            }
        )
    return summarize_fold_metrics(metric_rows), fold_rows, total_runtime


def dataset_detail(bench: dict, metrics: dict, method: str) -> dict:
    primary = bench["metrics"]["primary"]
    protocol = bench.get("protocol", {})
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Local canonical paper-inspired rerun {primary} {metrics[primary]:.4f}.",
        "protocol_note": (
            f"Canonical {protocol.get('type', 'protocol')} using a local {METHODS[method]['name']} inspired architecture. "
            "Official method code/checkpoints were not used."
        ),
    }


def device_metadata(device: torch.device) -> dict:
    metadata = {"device": str(device), "torch_version": torch.__version__}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata.update({"cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(index)})
    return metadata


def make_entry(method: str, benchmark_id: str, metrics: dict, fold_rows: list[dict], runtime: float, args, device: torch.device) -> dict:
    bench = load_benchmark(benchmark_id)
    spec = METHODS[method]
    primary = bench["metrics"]["primary"]
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "source": "local_paper_inspired_rerun",
        "protocol_status": "canonical",
        "runtime_seconds": round(runtime, 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "official_code_used": False,
        "implementation_variant": spec["variant"],
        **device_metadata(device),
    }
    if "Leave-One-Subject-Out" in bench.get("protocol", {}).get("type", ""):
        result["folds"] = len(fold_rows)
        result["fold_metrics"] = fold_rows

    return {
        "id": method,
        "method_name": spec["name"],
        "method_family": spec["family"],
        "submission_type": "platform",
        "coverage_status": "canonical_paper_inspired_rerun",
        "summary": f"{spec['name']} with local paper-inspired canonical reruns; these rows are not official author-code reproductions.",
        "implementation": spec["variant"],
        "har_scope": "Raw-window smartphone/wearable HAR under the site's canonical preprocessing protocol.",
        "datasets": [dataset_detail(bench, metrics, method)],
        "results": [result],
        "sources": sorted(set([bench.get("dataset_url", ""), *spec["sources"]])),
    }


def run_one(method: str, benchmark_id: str, args, device: torch.device) -> None:
    seed_all(args.seed)
    metrics, fold_rows, runtime = evaluate_dataset(method, benchmark_id, args, device)
    out_dir = RUNS_DIR / f"{benchmark_id}_{method}"
    write_json(out_dir / "results.json", [make_entry(method, benchmark_id, metrics, fold_rows, runtime, args, device)])
    primary = load_benchmark(benchmark_id)["metrics"]["primary"]
    print(f"{benchmark_id} / {METHODS[method]['name']}: {primary}={metrics[primary]:.4f}, runtime={runtime:.1f}s")


def select_device(args) -> torch.device:
    wants_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if not wants_cuda:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but this PyTorch build cannot see CUDA.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"CUDA GPU index {args.gpu} is unavailable; found {torch.cuda.device_count()} device(s).")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    return device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)

    device = select_device(args)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    print(f"Using device: {device}")
    for benchmark_id in args.datasets:
        for method in args.methods:
            run_one(method, benchmark_id, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
