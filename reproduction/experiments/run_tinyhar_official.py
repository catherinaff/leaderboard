"""Run TinyHAR with the official ISWC22-HAR model code on canonical windows.

This adapter keeps the leaderboard's canonical dataset protocols but imports
and trains `TinyHAR_Model` from the official repository without modifying the
model implementation. It does not use the earlier local TinyHARLite fallback.

The defaults follow the runnable TinyHAR setting in the official repository's
`notebooks/model/Train model.ipynb`. The checkpoint-only notebook setting with
`cross_channel_aggregation_type="naive"` is not runnable at the pinned commit
because the current model code wires that fusion branch with incompatible
dimensions.
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from har_utils import DATASETS_DIR, RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, write_json


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "ISWC22-HAR"
OFFICIAL_COMMIT = "b84b89d09f6914fe93e82cde423e294042da2741"
OFFICIAL_REPO_URL = "https://github.com/teco-kit/ISWC22-HAR"
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


def import_official_model():
    if not OFFICIAL_REPO.is_dir():
        raise SystemExit(f"Official TinyHAR repository missing: {OFFICIAL_REPO}")
    sys.path.insert(0, str(OFFICIAL_REPO))
    from models.TinyHAR import TinyHAR_Model  # type: ignore

    return TinyHAR_Model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train_val_split(train_idx: np.ndarray, seed: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    train_positions = np.flatnonzero(train_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(train_positions)
    cutoff = max(1, int(len(train_positions) * ratio))
    train_only = np.zeros_like(train_idx, dtype=bool)
    val_only = np.zeros_like(train_idx, dtype=bool)
    train_only[train_positions[:cutoff]] = True
    val_only[train_positions[cutoff:]] = True
    if not val_only.any():
        val_only[train_positions[-1]] = True
        train_only[train_positions[-1]] = False
    return train_only, val_only


def normalize_fold(x: np.ndarray, train_val_idx: np.ndarray) -> np.ndarray:
    mean = x[train_val_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_val_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float64)


def make_loader(x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(np.expand_dims(x[idx], axis=1))
    y_tensor = torch.from_numpy(y[idx].astype(np.int64))
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_model(channels: int, classes: int, length: int, args) -> nn.Module:
    TinyHAR_Model = import_official_model()
    return TinyHAR_Model(
        (1, 1, length, channels),
        classes,
        filter_num=args.filter_num,
        cross_channel_interaction_type=args.cross_channel_interaction_type,
        cross_channel_aggregation_type=args.cross_channel_aggregation_type,
        temporal_info_interaction_type=args.temporal_info_interaction_type,
        temporal_info_aggregation_type=args.temporal_info_aggregation_type,
    ).double()


def evaluate_loader(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> tuple[float, dict]:
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).double()
            yb = yb.to(device, non_blocking=True).long()
            logits = model(xb)
            losses.append(float(loss_fn(logits, yb).detach().cpu()))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    return float(np.mean(losses)), metric_bundle(np.concatenate(trues), np.concatenate(preds))


def train_eval_fold(x: np.ndarray, y: np.ndarray, train_val_idx: np.ndarray, test_idx: np.ndarray, args, device: torch.device) -> tuple[dict, float, int]:
    seed_all(args.seed)
    train_idx, val_idx = train_val_split(train_val_idx, args.seed, args.train_val_ratio)
    x_norm = normalize_fold(x, train_val_idx)
    model = make_model(x_norm.shape[2], int(y.max() + 1), x_norm.shape[1], args).to(device)
    train_loader = make_loader(x_norm, y, train_idx, args.batch_size, True)
    val_loader = make_loader(x_norm, y, val_idx, args.batch_size, False)
    test_loader = make_loader(x_norm, y, test_idx, args.batch_size, False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    start = time.time()
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).double()
            yb = yb.to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_loss, val_metrics = evaluate_loader(model, val_loader, loss_fn, device)
        epochs_run = epoch
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"TinyHAR official epoch={epoch} train_loss={np.mean(losses):.4f} "
                f"val_loss={val_loss:.4f} val_f1_macro={val_metrics['f1_macro']:.4f}",
                flush=True,
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
                print(f"TinyHAR official early_stop epoch={epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _loss, metrics = evaluate_loader(model, test_loader, loss_fn, device)
    return metrics, time.time() - start, epochs_run


def evaluate_dataset(benchmark_id: str, args, device: torch.device) -> tuple[dict, list[dict], float, int]:
    x, y, subjects, meta = load_dataset(benchmark_id)
    bench = load_benchmark(benchmark_id)
    protocol_type = bench.get("protocol", {}).get("type", "")
    if "Leave-One-Subject-Out" not in protocol_type:
        train_val_idx, test_idx = fixed_subject_split(subjects, meta)
        metrics, runtime, epochs_run = train_eval_fold(x, y, train_val_idx, test_idx, args, device)
        return metrics, [
            {
                "split": "fixed_subject_split",
                "train_val_windows": int(train_val_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
                "epochs_run": epochs_run,
            }
        ], runtime, epochs_run

    fold_rows = []
    metric_rows = []
    total_runtime = 0.0
    max_epochs_run = 0
    for subject in sorted(set(int(s) for s in subjects)):
        test_idx = subjects == subject
        train_val_idx = ~test_idx
        print(f"TinyHAR official {benchmark_id} LOSO subject={subject}", flush=True)
        metrics, runtime, epochs_run = train_eval_fold(x, y, train_val_idx, test_idx, args, device)
        metric_rows.append(metrics)
        total_runtime += runtime
        max_epochs_run = max(max_epochs_run, epochs_run)
        fold_rows.append(
            {
                "test_subject": int(subject),
                "train_val_windows": int(train_val_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
                "epochs_run": epochs_run,
            }
        )
    return summarize_fold_metrics(metric_rows), fold_rows, total_runtime, max_epochs_run


def dataset_detail(bench: dict, metrics: dict) -> dict:
    primary = bench["metrics"]["primary"]
    protocol = bench.get("protocol", {})
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Official TinyHAR code adapter canonical rerun {primary} {metrics[primary]:.4f}.",
        "protocol_note": "Canonical benchmark split/window protocol; model code imported from the official ISWC22-HAR TinyHAR repository.",
    }


def device_metadata(device: torch.device) -> dict:
    metadata = {"device": str(device), "torch_version": torch.__version__}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata.update({"cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(index)})
    return metadata


def make_entry(benchmark_id: str, metrics: dict, fold_rows: list[dict], runtime: float, epochs_run: int, args, device: torch.device) -> dict:
    bench = load_benchmark(benchmark_id)
    primary = bench["metrics"]["primary"]
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "source": "official_code_adapter",
        "protocol_status": "canonical",
        "runtime_seconds": round(runtime, 3),
        "epochs": args.epochs,
        "epochs_run": epochs_run,
        "seed": args.seed,
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": OFFICIAL_COMMIT,
        "official_model_class": "models.TinyHAR.TinyHAR_Model",
        "official_adapter_note": "Only the data adapter is leaderboard-specific; the TinyHAR model implementation is imported from the official repository. Defaults follow the runnable TinyHAR setting in notebooks/model/Train model.ipynb.",
        "official_configuration_source": "experiments/repos/ISWC22-HAR/notebooks/model/Train model.ipynb",
        "filter_num": args.filter_num,
        "cross_channel_interaction_type": args.cross_channel_interaction_type,
        "cross_channel_aggregation_type": args.cross_channel_aggregation_type,
        "temporal_info_interaction_type": args.temporal_info_interaction_type,
        "temporal_info_aggregation_type": args.temporal_info_aggregation_type,
        **device_metadata(device),
    }
    if "Leave-One-Subject-Out" in bench.get("protocol", {}).get("type", ""):
        result["folds"] = len(fold_rows)
        result["fold_metrics"] = fold_rows
    return {
        "id": "tinyhar",
        "method_name": "TinyHAR",
        "year": 2022,
        "method_family": "Lightweight CNN/RNN for edge HAR",
        "submission_type": "platform",
        "coverage_status": "canonical_official_code_adapter",
        "summary": "TinyHAR official repository model code run against the leaderboard's canonical HAR benchmark windows.",
        "implementation": "Imports TinyHAR_Model from github.com/teco-kit/ISWC22-HAR and trains it with the official Adam + CrossEntropy training pattern.",
        "har_scope": "Resource-constrained wearable HAR and TinyML-oriented deployment.",
        "datasets": [dataset_detail(bench, metrics)],
        "results": [result],
        "sources": [
            "https://github.com/teco-kit/ISWC22-HAR",
            "https://publikationen.bibliothek.kit.edu/1000150216/149250239",
            bench.get("dataset_url", ""),
        ],
    }


def run_one(benchmark_id: str, args, device: torch.device) -> None:
    seed_all(args.seed)
    metrics, fold_rows, runtime, epochs_run = evaluate_dataset(benchmark_id, args, device)
    out_dir = RUNS_DIR / f"{benchmark_id}_tinyhar_official"
    write_json(out_dir / "results.json", [make_entry(benchmark_id, metrics, fold_rows, runtime, epochs_run, args, device)])
    primary = load_benchmark(benchmark_id)["metrics"]["primary"]
    print(f"{benchmark_id} / TinyHAR official: {primary}={metrics[primary]:.4f}, runtime={runtime:.1f}s")


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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-val-ratio", type=float, default=0.90)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--filter-num", type=int, default=20)
    parser.add_argument("--cross-channel-interaction-type", default="attn", choices=["attn", "transformer", "identity", "FCinter"])
    parser.add_argument("--cross-channel-aggregation-type", default="FC", choices=["filter", "naive", "FC", "FCnaive", "naive2", "SFCF", "SFCF2", "SFCC"])
    parser.add_argument("--temporal-info-interaction-type", default="lstm", choices=["gru", "lstm", "attn", "transformer", "identity", "conv"])
    parser.add_argument("--temporal-info-aggregation-type", default="tnaive", choices=["filter", "naive", "tnaive", "FC", "identiry"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)

    if not 0.0 < args.train_val_ratio < 1.0:
        raise SystemExit("--train-val-ratio must be between 0 and 1")
    device = select_device(args)
    print(f"Using device: {device}")
    print(f"Using official TinyHAR repo: {OFFICIAL_REPO} @ {OFFICIAL_COMMIT}")
    for benchmark_id in args.datasets:
        run_one(benchmark_id, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
