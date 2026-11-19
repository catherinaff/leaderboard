"""SensorLLM paper-protocol proxy rerun.

The public paper reports SensorLLM HAR results, but the full method depends on
LLM/Chronos components and checkpoints that are not available in this local
workspace. This runner keeps the dataset protocols aligned with the paper and
uses a compact sensor-token neural proxy to quantify the gap before any
leaderboard promotion.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "experiments" / "datasets"
RUNS_DIR = ROOT / "experiments" / "runs"
GAP_THRESHOLD = 0.05

UCI_SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

PAMAP2_ACTIVITIES = [1, 2, 3, 17, 16, 12, 13, 4, 7, 6, 5, 24]
PAMAP2_LABEL = {activity: i for i, activity in enumerate(PAMAP2_ACTIVITIES)}

PAPER_REFERENCES = {
    "uci_har": {
        "benchmark_id": "uci_har",
        "dataset": "UCI-HAR",
        "classes": 6,
        "paper_metrics": {"accuracy": 0.908, "f1_macro": 0.912},
        "primary_metric": "accuracy",
        "source": "SensorLLM paper Table 2, UCI-HAR row: F1-macro/Accuracy 91.2/90.8%.",
        "protocol_note": "UCI-HAR fixed 70/30 split, 6 inertial channels, 50Hz, 128-sample windows, 64-sample stride.",
    },
    "mhealth": {
        "benchmark_id": "mhealth",
        "dataset": "MHEALTH",
        "classes": 12,
        "paper_metrics": {"accuracy": 0.890, "f1_macro": 0.894},
        "primary_metric": "accuracy",
        "source": "SensorLLM paper Table 2, MHEALTH row: F1-macro/Accuracy 89.4/89.0%.",
        "protocol_note": "MHEALTH paper protocol: test subjects 1, 3, and 6; 15 acceleration/gyroscope channels; 50Hz, 100-sample windows, 50-sample stride.",
    },
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "classes": 12,
        "paper_metrics": {"accuracy": 0.872, "f1_macro": 0.862},
        "primary_metric": "accuracy",
        "source": "SensorLLM paper Table 2, PAMAP2 row: F1-macro/Accuracy 86.2/87.2%.",
        "protocol_note": "PAMAP2 paper protocol: test subjects 105 and 106; 27 IMU acc/gyro/magnetometer channels; downsample 100Hz to 50Hz, 100-sample windows, 50-sample stride.",
    },
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    rows = np.arange(out.shape[0])
    for col in range(out.shape[1]):
        x = out[:, col]
        good = np.isfinite(x)
        if good.all():
            continue
        if not good.any():
            x[:] = 0.0
        else:
            x[~good] = np.interp(rows[~good], rows[good], x[good])
    return out


def sliding_windows(values: np.ndarray, labels: np.ndarray, window: int, stride: int) -> tuple[list[np.ndarray], list[int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    if len(values) < window:
        return xs, ys
    for start in range(0, len(values) - window + 1, stride):
        end = start + window
        label_counts = np.bincount(labels[start:end].astype(np.int64))
        xs.append(values[start:end].astype(np.float32))
        ys.append(int(label_counts.argmax()))
    return xs, ys


def normalize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def load_uci_har() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "uci_har" / "UCI HAR Dataset"
    splits = {}
    for split in ["train", "test"]:
        sig_dir = data_dir / split / "Inertial Signals"
        arrays = [np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32) for name in UCI_SIGNALS]
        x = np.stack(arrays, axis=-1)
        y = np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=np.int64) - 1
        splits[split] = (x, y)
    x_train, y_train = splits["train"]
    x_test, y_test = splits["test"]
    x_train, x_test = normalize_train_test(x_train, x_test)
    meta = {
        "raw_dir": str(data_dir),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "window_size_samples": 128,
        "stride_samples": 64,
        "channels": "6 raw inertial signals: body_acc_xyz and body_gyro_xyz.",
        "split": "Official UCI-HAR fixed train/test split.",
    }
    return x_train, y_train, x_test, y_test, meta


def load_mhealth() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "mhealth" / "MHEALTHDATASET"
    if not data_dir.is_dir():
        data_dir = DATASETS_DIR / "mhealth"
    # chest acc; left ankle acc+gyro; right lower arm acc+gyro.
    cols = [0, 1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19]
    test_subjects = {1, 3, 6}
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    file_rows = []
    for path in sorted(data_dir.glob("mHealth_subject*.log")):
        match = re.search(r"subject(\d+)", path.name)
        subject = int(match.group(1)) if match else 0
        raw = np.loadtxt(path, dtype=np.float32)
        values = raw[:, cols]
        labels = raw[:, 23].astype(np.int64)
        keep = labels > 0
        values = values[keep]
        labels = labels[keep] - 1
        values = interpolate_nan(values)
        start = 0
        windows = 0
        while start < len(values):
            label = int(labels[start])
            end = start + 1
            while end < len(values) and int(labels[end]) == label:
                end += 1
            xs, ys = sliding_windows(values[start:end], labels[start:end], 100, 50)
            if subject in test_subjects:
                test_x.extend(xs)
                test_y.extend(ys)
            else:
                train_x.extend(xs)
                train_y.extend(ys)
            windows += len(ys)
            start = end
        file_rows.append({"file": path.name, "subject": subject, "windows": windows, "split": "test" if subject in test_subjects else "train"})
    x_train = np.asarray(train_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int64)
    x_test = np.asarray(test_x, dtype=np.float32)
    y_test = np.asarray(test_y, dtype=np.int64)
    x_train, x_test = normalize_train_test(x_train, x_test)
    meta = {
        "raw_dir": str(data_dir),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "window_size_samples": 100,
        "stride_samples": 50,
        "channels": "15 channels: chest acc, left-ankle acc+gyro, right-lower-arm acc+gyro.",
        "split": "Train subjects 2,4,5,7,8,9,10; test subjects 1,3,6.",
        "file_rows": file_rows,
    }
    return x_train, y_train, x_test, y_test, meta


def pamap2_cols_27() -> list[int]:
    cols: list[int] = []
    for base in [3, 20, 37]:
        cols.extend(range(base + 1, base + 4))
        cols.extend(range(base + 7, base + 13))
    return cols


def load_pamap2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "pamap2" / "PAMAP2_Dataset" / "Protocol"
    test_subjects = {105, 106}
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    cols = pamap2_cols_27()
    file_rows = []
    for path in sorted(data_dir.glob("subject*.dat")):
        subject = int(path.stem.replace("subject", ""))
        raw = np.loadtxt(path, dtype=np.float32)
        # Downsample from 100Hz to 50Hz before windowing.
        raw = raw[::2]
        activity = raw[:, 1].astype(np.int64)
        values = interpolate_nan(raw[:, cols])
        start = 0
        windows = 0
        while start < len(values):
            act = int(activity[start])
            end = start + 1
            while end < len(values) and int(activity[end]) == act:
                end += 1
            if act in PAMAP2_LABEL:
                labels = np.full(end - start, PAMAP2_LABEL[act], dtype=np.int64)
                xs, ys = sliding_windows(values[start:end], labels, 100, 50)
                if subject in test_subjects:
                    test_x.extend(xs)
                    test_y.extend(ys)
                else:
                    train_x.extend(xs)
                    train_y.extend(ys)
                windows += len(ys)
            start = end
        file_rows.append({"file": path.name, "subject": subject, "windows": windows, "split": "test" if subject in test_subjects else "train"})
    x_train = np.asarray(train_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int64)
    x_test = np.asarray(test_x, dtype=np.float32)
    y_test = np.asarray(test_y, dtype=np.int64)
    x_train, x_test = normalize_train_test(x_train, x_test)
    meta = {
        "raw_dir": str(data_dir),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "window_size_samples": 100,
        "stride_samples": 50,
        "channels": "27 channels: per IMU acc16_xyz, gyro_xyz, magnetometer_xyz; HR, temperature, acc6, and orientation removed.",
        "split": "Train subjects except 105/106; test subjects 105 and 106.",
        "downsample": "raw[::2] from 100Hz to 50Hz.",
        "file_rows": file_rows,
    }
    return x_train, y_train, x_test, y_test, meta


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if name == "uci_har":
        return load_uci_har()
    if name == "mhealth":
        return load_mhealth()
    if name == "pamap2":
        return load_pamap2()
    raise KeyError(name)


class SensorProxyNet(nn.Module):
    def __init__(self, channels: int, classes: int, window: int, d_model: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.stats = nn.Sequential(
            nn.Linear(channels * 5, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,C
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        minv = x.amin(dim=1)
        maxv = x.amax(dim=1)
        slope = x[:, -1, :] - x[:, 0, :]
        stats = self.stats(torch.cat([mean, std, minv, maxv, slope], dim=1))
        h = self.input(x.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(h + self.pos[:, : h.shape[1], :])
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1), stats], dim=1)
        return self.head(pooled)


def make_split(y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def class_weights(y: np.ndarray, classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> dict:
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size)
    pred = []
    model.eval()
    for xb, _ in loader:
        pred.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(pred)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def run_trial(dataset: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, args, seed: int, device: torch.device) -> dict:
    seed_all(seed)
    ref = PAPER_REFERENCES[dataset]
    train_idx, val_idx = make_split(y_train, args.val_fraction, seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train[train_idx]), torch.from_numpy(y_train[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SensorProxyNet(x_train.shape[2], ref["classes"], x_train.shape[1], d_model=args.d_model, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    weights = class_weights(y_train[train_idx], ref["classes"], device)
    best_val = -1.0
    best_state = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=weights, label_smoothing=args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, x_train[val_idx], y_train[val_idx], args.eval_batch_size, device)
            score = val_metrics["f1_macro"]
            if score > best_val:
                best_val = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if args.verbose:
                print(f"{dataset} seed={seed} epoch={epoch} loss={np.mean(losses):.4f} val_f1_macro={score:.4f} best={best_val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, x_test, y_test, args.eval_batch_size, device)
    return {
        "seed": seed,
        "runtime_seconds": round(time.time() - start, 3),
        "best_val_f1_macro": float(best_val),
        "test": test_metrics,
    }


def summarize_dataset(dataset: str, trials: list[dict], meta: dict, args) -> tuple[dict, dict]:
    ref = PAPER_REFERENCES[dataset]
    metrics = {}
    for key in ["accuracy", "f1_macro", "f1_weighted"]:
        vals = np.asarray([t["test"][key] for t in trials], dtype=np.float64)
        metrics[key] = float(vals.mean())
        metrics[f"{key}_std"] = float(vals.std(ddof=0))
    primary_metric = ref["primary_metric"]
    primary_score = metrics[primary_metric]
    paper_score = ref["paper_metrics"][primary_metric]
    gap_signed = primary_score - paper_score
    gap_abs = abs(gap_signed)
    status = "passed" if gap_abs <= args.gap_threshold else "failed"
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": primary_score,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "source": "paper_protocol_proxy_rerun",
        "protocol_status": "paper_protocol_reimplementation_proxy",
        "official_code_used": False,
        "paper_reference_score": paper_score,
        "paper_reference_metric": primary_metric,
        "paper_reference_source": ref["source"],
        "paper_reference_gap_signed": gap_signed,
        "paper_reference_gap_abs": gap_abs,
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range" if status == "passed" else "outside_expected_range",
        "calibration_status": status,
        "paper_metrics": ref["paper_metrics"],
        "trials": trials,
        "proxy_note": "Compact sensor-token Transformer proxy. This is not an official SensorLLM LLaMA/Chronos rerun.",
    }
    dataset_entry = {
        "benchmark_id": ref["benchmark_id"],
        "dataset": ref["dataset"],
        "reported_effect": f"SensorLLM proxy rerun {primary_metric} {primary_score:.4f} vs paper {paper_score:.4f}; gap {gap_abs:.4f}, {status}.",
        "protocol_note": ref["protocol_note"],
        "metadata": meta,
    }
    return result, dataset_entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_REFERENCES), default=["uci_har", "mhealth"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--run-tag", default="sensorllm_paper_proxy_20260727")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    results = []
    datasets = []
    for dataset in args.datasets:
        x_train, y_train, x_test, y_test, meta = load_dataset(dataset)
        print(f"{dataset}: train={x_train.shape} test={x_test.shape} classes={len(set(y_train.tolist()) | set(y_test.tolist()))}", flush=True)
        trials = [run_trial(dataset, x_train, y_train, x_test, y_test, args, seed, device) for seed in args.seeds]
        result, dataset_entry = summarize_dataset(dataset, trials, meta, args)
        results.append(result)
        datasets.append(dataset_entry)
        print(
            f"{dataset} {result['primary_metric']}={result['primary_score']:.6f} "
            f"paper={result['paper_reference_score']:.6f} gap={result['paper_reference_gap_abs']:.6f} "
            f"{result['calibration_status']}",
            flush=True,
        )

    coverage = "paper_protocol_calibration_proxy" if results and all(r["calibration_status"] == "passed" for r in results) else "paper_protocol_proxy_partial_or_failed"
    payload = [
        {
            "id": "sensorllm",
            "method_name": "SensorLLM",
            "year": 2025,
            "method_family": "sensor-language LLM HAR",
            "submission_type": "paper_protocol_proxy_rerun",
            "paper_url": "https://arxiv.org/abs/2410.10624",
            "code_url": "https://github.com/cruiseresearchgroup/SensorLLM",
            "coverage_status": coverage,
            "summary": "SensorLLM paper-protocol proxy rerun against Table 2 scores.",
            "implementation": "Local compact sensor-token Transformer proxy. Official SensorLLM LLaMA/Chronos code and checkpoints were not available locally during this run.",
            "har_scope": "Wearable HAR paper-protocol calibration proxy before any official-code promotion.",
            "datasets": datasets,
            "results": results,
            "sources": [
                "https://arxiv.org/abs/2410.10624",
                "https://aclanthology.org/2025.emnlp-main.19.pdf",
                "https://github.com/cruiseresearchgroup/SensorLLM",
            ],
        }
    ]
    out = RUNS_DIR / args.run_tag / "results.json"
    write_json(out, payload)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
