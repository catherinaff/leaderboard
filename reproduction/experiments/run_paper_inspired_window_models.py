"""Run paper-inspired PyTorch HAR models on processed window datasets."""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from har_utils import RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, write_json


METHODS = {
    "deepconvlstm": {
        "name": "DeepConvLSTM",
        "family": "CNN + LSTM",
        "source": "local_deepconvlstm_style_rerun",
    },
    "dcnn": {
        "name": "DCNN",
        "family": "Deep CNN",
        "source": "local_paper_inspired_rerun",
    },
    "blstm_s": {
        "name": "b-LSTM-S",
        "family": "Bidirectional LSTM",
        "source": "local_paper_inspired_rerun",
    },
    "attention_model": {
        "name": "Attention Model",
        "family": "CNN/GRU + Attention",
        "source": "local_paper_inspired_rerun",
    },
}


class DCNN(nn.Module):
    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).squeeze(-1))


class DeepConvLSTM(nn.Module):
    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        layers = []
        in_ch = channels
        for _ in range(4):
            layers.extend([nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.ReLU()])
            in_ch = 64
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.5)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


class BiLSTMS(nn.Module):
    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(channels, 128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(256, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.transpose(1, 2))
        return self.fc(out[:, -1, :])


class AttentionHAR(nn.Module):
    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)
        self.fc = nn.Linear(256, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).transpose(1, 2)
        out, _ = self.gru(x)
        weights = torch.softmax(self.attn(out), dim=1)
        return self.fc((weights * out).sum(dim=1))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(method: str, channels: int, classes: int) -> nn.Module:
    if method == "deepconvlstm":
        return DeepConvLSTM(channels, classes)
    if method == "dcnn":
        return DCNN(channels, classes)
    if method == "blstm_s":
        return BiLSTMS(channels, classes)
    if method == "attention_model":
        return AttentionHAR(channels, classes)
    raise ValueError(method)


def normalize_split(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def train_eval(method: str, x_train, y_train, x_test, y_test, args, device) -> tuple[dict, float]:
    x_train, x_test = normalize_split(x_train, x_test)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    classes = int(max(y_train.max(), y_test.max()) + 1)
    model = make_model(method, x_train.shape[1], classes).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    pred = np.concatenate(preds)
    return metric_bundle(y_test, pred), time.time() - start


def fixed_subject_split(subjects: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    train_subjects = set(int(s) for s in meta.get("train_subjects", []))
    test_subjects = set(int(s) for s in meta.get("test_subjects", []))
    if not train_subjects or not test_subjects:
        raise SystemExit("Fixed split dataset requires train_subjects and test_subjects in processed metadata.")
    return np.asarray([int(s) in train_subjects for s in subjects]), np.asarray([int(s) in test_subjects for s in subjects])


def evaluate_dataset(method: str, benchmark_id: str, args, device) -> tuple[dict, list[dict], float]:
    x, y, subjects, meta = load_windows(benchmark_id)
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


def dataset_detail(bench: dict, metrics: dict) -> dict:
    primary = bench["metrics"]["primary"]
    protocol = bench.get("protocol", {})
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Local canonical paper-inspired rerun {primary} {metrics[primary]:.4f}.",
        "protocol_note": f"Canonical {protocol.get('type', 'protocol')} using local PyTorch paper-inspired architecture, {protocol.get('window_size_samples')} sample windows, and {protocol.get('stride_samples')} sample stride.",
    }


def device_metadata(device: torch.device) -> dict:
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(index),
            }
        )
    return metadata


def entry(method: str, benchmark_id: str, metrics: dict, fold_rows: list[dict], runtime: float, args, device: torch.device) -> dict:
    bench = load_benchmark(benchmark_id)
    spec = METHODS[method]
    primary = bench["metrics"]["primary"]
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "source": spec["source"],
        "protocol_status": "canonical",
        "runtime_seconds": round(runtime, 3),
        "epochs": args.epochs,
        "seed": args.seed,
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
        "summary": f"{spec['name']} paper-inspired local implementation trained on canonical {bench['name']} windows.",
        "implementation": "PyTorch local implementation using the paper family architecture rather than an official author checkpoint.",
        "har_scope": "Raw-window wearable/smartphone HAR under the site's canonical preprocessing protocol.",
        "datasets": [dataset_detail(bench, metrics)],
        "results": [result],
        "sources": [bench.get("dataset_url", "")],
    }


def run_one(method: str, benchmark_id: str, args, device) -> None:
    seed_all(args.seed)
    metrics, fold_rows, runtime = evaluate_dataset(method, benchmark_id, args, device)
    out_dir = RUNS_DIR / f"{benchmark_id}_{method}"
    write_json(out_dir / "results.json", [entry(method, benchmark_id, metrics, fold_rows, runtime, args, device)])
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
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
