"""Run IF-ConvTransformer official PyTorch model code on canonical windows."""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from har_utils import DATASETS_DIR, RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, write_json


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "IF-ConvTransformer-UbiComp2022"
OFFICIAL_COMMIT = "577fa56"
OFFICIAL_REPO_URL = "https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022"
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


class LabelSmoothingCrossEntropy(nn.Module):
    """Same loss family used by the official IF-ConvTransformer train_op."""

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = output.size(-1)
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        smooth_loss = -log_preds.sum(dim=-1) / classes
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction="none")
        return (self.eps * smooth_loss + (1.0 - self.eps) * nll).mean()


def install_import_shims() -> None:
    """Shim unused optional imports so the official model file can be imported unchanged."""
    sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        for name in ["figure", "plot", "savefig", "close", "legend", "xlabel", "ylabel", "title", "show"]:
            setattr(pyplot, name, lambda *args, **kwargs: None)
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot
    if "contiguous_params" not in sys.modules:
        contig = types.ModuleType("contiguous_params")

        class ContiguousParams:
            def __init__(self, params):
                self._params = list(params)

            def contiguous(self):
                return self._params

        contig.ContiguousParams = ContiguousParams
        sys.modules["contiguous_params"] = contig
    if "utils.utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_mod = types.ModuleType("utils.utils")
        utils_mod.LabelSmoothingCrossEntropy = LabelSmoothingCrossEntropy
        sys.modules["utils"] = utils_pkg
        sys.modules["utils.utils"] = utils_mod


def import_official_model():
    src = OFFICIAL_REPO / "src"
    if not src.is_dir():
        raise SystemExit(f"Official IF-ConvTransformer repository missing or incomplete: {src}")
    install_import_shims()
    sys.path.insert(0, str(src))
    from classifiers.If_ConvTransformer_torch import If_ConvTransformer  # type: ignore

    return If_ConvTransformer


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


def train_val_split(train_idx: np.ndarray, y: np.ndarray, seed: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
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


def adapt_to_if9(x: np.ndarray, benchmark_id: str) -> tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=np.float32)
    channels = x.shape[2]
    out = np.zeros((x.shape[0], x.shape[1], 9), dtype=np.float32)
    if benchmark_id == "uci_har" and channels >= 9:
        body_acc = x[:, :, :3]
        body_gyro = x[:, :, 3:6]
        total_acc = x[:, :, 6:9]
        out[:, :, 0:3] = total_acc - body_acc
        out[:, :, 3:6] = body_gyro
        out[:, :, 6:9] = body_acc
        return out[:, None, :, :].transpose(0, 1, 3, 2), "uci_gravity_total_minus_body__gyro__body_acc_to_9ch"
    if channels == 3:
        out[:, :, 6:9] = x
        return out[:, None, :, :].transpose(0, 1, 3, 2), "acc_only_zero_gravity_gyro_to_9ch"
    if channels == 6:
        out[:, :, 3:6] = x[:, :, 3:6]
        out[:, :, 6:9] = x[:, :, 0:3]
        return out[:, None, :, :].transpose(0, 1, 3, 2), "acc_gyro_zero_gravity_to_9ch"
    out[:, :, :] = x[:, :, :9]
    return out[:, None, :, :].transpose(0, 1, 3, 2), f"first9_of_{channels}ch"


def normalize_fold(x: np.ndarray, train_val_idx: np.ndarray) -> np.ndarray:
    mean = x[train_val_idx].mean(axis=(0, 3), keepdims=True)
    std = x[train_val_idx].std(axis=(0, 3), keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(x[idx])
    y_tensor = torch.from_numpy(y[idx].astype(np.int64))
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_model(length: int, classes: int, args) -> nn.Module:
    IfConvTransformer = import_official_model()
    return IfConvTransformer(
        1,
        9,
        args.feature_channel,
        args.kernel_size,
        args.kernel_size_grav,
        args.scale_num,
        args.feature_channel_out,
        args.multiheads,
        args.drop_rate,
        "Canonical",
        length,
        classes,
    )


def evaluate_loader(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> tuple[float, dict]:
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            logits = model(xb)[0]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            losses.append(float(loss_fn(logits, yb).detach().cpu()))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    return float(np.mean(losses)), metric_bundle(np.concatenate(trues), np.concatenate(preds))


def train_eval_fold(x: np.ndarray, y: np.ndarray, train_val_idx: np.ndarray, test_idx: np.ndarray, args, device: torch.device) -> tuple[dict, float]:
    seed_all(args.seed)
    train_idx, val_idx = train_val_split(train_val_idx, y, args.seed, args.train_val_ratio)
    x_norm = normalize_fold(x, train_val_idx)
    model = make_model(x_norm.shape[-1], int(y.max() + 1), args).to(device)
    train_loader = make_loader(x_norm, y, train_idx, args.batch_size, True)
    val_loader = make_loader(x_norm, y, val_idx, args.batch_size, False)
    test_loader = make_loader(x_norm, y, test_idx, args.batch_size, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, min_lr=args.lr / 10)
    loss_fn = LabelSmoothingCrossEntropy(args.label_smoothing).to(device)

    best_state = None
    best_val_accuracy = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)[0]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss, val_metrics = evaluate_loader(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"IF-ConvTransformer official epoch={epoch} train_loss={np.mean(train_losses):.4f} "
                f"val_loss={val_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    _loss, metrics = evaluate_loader(model, test_loader, loss_fn, device)
    return metrics, time.time() - start


def evaluate_dataset(benchmark_id: str, args, device: torch.device) -> tuple[dict, list[dict], float, str]:
    x_raw, y, subjects, meta = load_dataset(benchmark_id)
    x, adapter_note = adapt_to_if9(x_raw, benchmark_id)
    bench = load_benchmark(benchmark_id)
    protocol_type = bench.get("protocol", {}).get("type", "")
    if "Leave-One-Subject-Out" not in protocol_type:
        train_val_idx, test_idx = fixed_subject_split(subjects, meta)
        metrics, runtime = train_eval_fold(x, y, train_val_idx, test_idx, args, device)
        return metrics, [
            {
                "split": "fixed_subject_split",
                "train_val_windows": int(train_val_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
            }
        ], runtime, adapter_note
    fold_rows = []
    metric_rows = []
    total_runtime = 0.0
    for subject in sorted(set(int(s) for s in subjects)):
        test_idx = subjects == subject
        train_val_idx = ~test_idx
        print(f"IF-ConvTransformer official {benchmark_id} LOSO subject={subject}", flush=True)
        metrics, runtime = train_eval_fold(x, y, train_val_idx, test_idx, args, device)
        metric_rows.append(metrics)
        total_runtime += runtime
        fold_rows.append(
            {
                "test_subject": int(subject),
                "train_val_windows": int(train_val_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
            }
        )
    return summarize_fold_metrics(metric_rows), fold_rows, total_runtime, adapter_note


def dataset_detail(bench: dict, metrics: dict, adapter_note: str) -> dict:
    primary = bench["metrics"]["primary"]
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Official IF-ConvTransformer model-code adapter canonical rerun {primary} {metrics[primary]:.4f}.",
        "protocol_note": f"Canonical benchmark split/window protocol; official smartphone IF-ConvTransformer 9-channel input adapter: {adapter_note}.",
    }


def device_metadata(device: torch.device) -> dict:
    metadata = {"device": str(device), "torch_version": torch.__version__}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata.update({"cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(index)})
    return metadata


def make_entry(benchmark_id: str, metrics: dict, fold_rows: list[dict], runtime: float, adapter_note: str, args, device: torch.device) -> dict:
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
        "seed": args.seed,
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": OFFICIAL_COMMIT,
        "official_model_class": "classifiers.If_ConvTransformer_torch.If_ConvTransformer",
        "official_configuration_source": "README HAPT training command and src/utils/constants.py create_classifier defaults",
        "official_adapter_note": "Official model code is imported unchanged; unused repo utility/plot imports are shimmed by the adapter.",
        "architecture_variant": "smartphone_if_convtransformer",
        "input_channel_adapter": adapter_note,
        "normalization": "per-fold z-score using train+validation windows",
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "label_smoothing": args.label_smoothing,
        "feature_channel": args.feature_channel,
        "feature_channel_out": args.feature_channel_out,
        "scale_num": args.scale_num,
        **device_metadata(device),
    }
    if "Leave-One-Subject-Out" in bench.get("protocol", {}).get("type", ""):
        result["folds"] = len(fold_rows)
        result["fold_metrics"] = fold_rows
    return {
        "id": "if_convtransformer",
        "method_name": "IF-ConvTransformer",
        "year": 2022,
        "method_family": "IMU fusion ConvTransformer",
        "submission_type": "platform",
        "coverage_status": "canonical_official_code_adapter",
        "summary": "IF-ConvTransformer official PyTorch model code run against the leaderboard's canonical HAR benchmark windows.",
        "implementation": "Imports If_ConvTransformer from github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022 and trains it with the official Adam + label smoothing pattern.",
        "har_scope": "IMU fusion block plus ConvTransformer backbone for smartphone and wearable HAR.",
        "datasets": [dataset_detail(bench, metrics, adapter_note)],
        "results": [result],
        "sources": [
            OFFICIAL_REPO_URL,
            "https://dl.acm.org/doi/pdf/10.1145/3534584",
            bench.get("dataset_url", ""),
        ],
    }


def run_one(benchmark_id: str, args, device: torch.device) -> None:
    seed_all(args.seed)
    metrics, fold_rows, runtime, adapter_note = evaluate_dataset(benchmark_id, args, device)
    out_dir = RUNS_DIR / f"{benchmark_id}_if_convtransformer_official"
    write_json(out_dir / "results.json", [make_entry(benchmark_id, metrics, fold_rows, runtime, adapter_note, args, device)])
    primary = load_benchmark(benchmark_id)["metrics"]["primary"]
    print(f"{benchmark_id} / IF-ConvTransformer official: {primary}={metrics[primary]:.4f}, runtime={runtime:.1f}s", flush=True)


def select_device(args) -> torch.device:
    wants_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if not wants_cuda:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but this PyTorch build cannot see CUDA.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    return device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-val-ratio", type=float, default=0.90)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--feature-channel", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--kernel-size-grav", type=int, default=3)
    parser.add_argument("--scale-num", type=int, default=2)
    parser.add_argument("--feature-channel-out", type=int, default=128)
    parser.add_argument("--multiheads", type=int, default=1)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)
    if not 0.0 < args.train_val_ratio < 1.0:
        raise SystemExit("--train-val-ratio must be between 0 and 1")
    device = select_device(args)
    print(f"Using device: {device}", flush=True)
    print(f"Using official IF-ConvTransformer repo: {OFFICIAL_REPO} @ {OFFICIAL_COMMIT}", flush=True)
    for benchmark_id in args.datasets:
        run_one(benchmark_id, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
