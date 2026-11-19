"""Reimplement the Attention Model paper datasets and compare with paper values.

Murahari & Ploetz (2018) report mean F1 on OPPORTUNITY, PAMAP2, and Skoda.
No maintained official code/checkpoint is available in this workspace, so this
runner keeps the paper datasets/splits used by the Attend-And-Discriminate
coverage files and trains a DeepConvLSTM-style encoder with temporal attention.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
AD_REPO = ROOT / "experiments" / "repos" / "Attend-And-Discriminate"
GAP_THRESHOLD = 0.05

PAPER_REFERENCES = {
    "opportunity": {
        "benchmark_id": "opportunity_a",
        "dataset": "OPPORTUNITY",
        "classes": 18,
        "input_dim": 79,
        "paper_score": 0.707,
        "paper_metric": "f1_macro",
        "source": "Murahari & Ploetz 2018, On Attention Models for Human Activity Recognition, Table 1",
        "note": "Paper table reports mean F1 70.7 on OPPORTUNITY.",
    },
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "classes": 12,
        "input_dim": 52,
        "paper_score": 0.875,
        "paper_metric": "f1_macro",
        "source": "Murahari & Ploetz 2018, On Attention Models for Human Activity Recognition, Table 1",
        "note": "Paper table reports mean F1 87.5 on PAMAP2.",
    },
    "skoda": {
        "benchmark_id": "skoda",
        "dataset": "Skoda Mini Checkpoint",
        "classes": 11,
        "input_dim": 60,
        "paper_score": 0.913,
        "paper_metric": "f1_macro_no_null",
        "source": "Murahari & Ploetz 2018, On Attention Models for Human Activity Recognition, Table 1",
        "note": "Paper table reports mean F1 91.3 on Skoda; primary comparison excludes Null to match the 10-gesture wording used by the site.",
    },
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def sliding_window(x: np.ndarray, y: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        xs.append(x[start:end])
        ys.append(y[start:end][-1])
        start += stride
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def standardize(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
    )


def load_from_ad_processed(name: str, split: str, *, sample_wise_test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    filename = "test_sample_wise.npz" if split == "test" and sample_wise_test else f"{split}.npz"
    path = AD_REPO / "data" / name / "processed" / "24_12" / filename
    data = np.load(path)
    return data["data"].astype(np.float32), data["target"].astype(np.int64)


def load_from_ad_mat(name: str, window: int, stride: int, stride_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    mat_path = AD_REPO / "dataset" / f"{name}.mat"
    contents = sio.loadmat(mat_path)
    if name == "opportunity":
        x_train = contents["trainingData"].astype(np.float32).T
        y_train = contents["trainingLabels"].reshape(-1).astype(np.int64) - 1
        x_val = contents["valData"].astype(np.float32).T
        y_val = contents["valLabels"].reshape(-1).astype(np.int64) - 1
        x_test = contents["testingData"].astype(np.float32).T
        y_test = contents["testingLabels"].reshape(-1).astype(np.int64) - 1
    else:
        x_train = contents["X_train"].astype(np.float32)
        y_train = contents["y_train"].reshape(-1).astype(np.int64)
        x_val = contents["X_valid"].astype(np.float32)
        y_val = contents["y_valid"].reshape(-1).astype(np.int64)
        x_test = contents["X_test"].astype(np.float32)
        y_test = contents["y_test"].reshape(-1).astype(np.int64)
    x_train, x_val, x_test = standardize(x_train, x_val, x_test)
    x_train, y_train = sliding_window(x_train, y_train, window, stride)
    x_val, y_val = sliding_window(x_val, y_val, window, stride)
    x_test, y_test = sliding_window(x_test, y_test, window, stride_test)
    return x_train, y_train, x_val, y_val, x_test, y_test, {"mat_path": str(mat_path), "stride_test_samples": stride_test}


def load_dataset(name: str, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    meta = {
        "source_repo": str(AD_REPO),
        "window_size_samples": args.window,
        "stride_samples": args.stride,
        "split_source": "Attend-And-Discriminate paper-coverage dataset files",
    }
    if name in {"pamap2", "skoda"} and not args.force_mat:
        x_train, y_train = load_from_ad_processed(name, "train")
        x_val, y_val = load_from_ad_processed(name, "val")
        x_test, y_test = load_from_ad_processed(name, "test", sample_wise_test=args.sample_wise_test)
        meta["loader"] = "preprocessed_npz_24_12"
        if args.sample_wise_test:
            meta["test_loader"] = "test_sample_wise.npz"
    else:
        stride_test = args.stride_test if args.sample_wise_test else args.stride
        x_train, y_train, x_val, y_val, x_test, y_test, mat_meta = load_from_ad_mat(name, args.window, args.stride, stride_test)
        meta.update(mat_meta)
        meta["loader"] = "mat_rewindowed"
    return x_train, y_train, x_val, y_val, x_test, y_test, meta


class AttentionDeepConvLSTM(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        *,
        filters: int = 64,
        hidden: int = 128,
        dropout: float = 0.5,
        dropout_rnn: float = 0.25,
        dropout_cls: float = 0.5,
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_ch = 1
        for _ in range(4):
            conv_layers.extend([nn.Conv2d(in_ch, filters, kernel_size=(5, 1)), nn.ReLU()])
            in_ch = filters
        self.conv = nn.Sequential(*conv_layers)
        self.drop_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters * channels, hidden, num_layers=2, batch_first=True, dropout=dropout_rnn)
        self.attn_hidden = nn.Linear(hidden, hidden)
        self.attn_score = nn.Linear(hidden, 1)
        self.drop_out = nn.Dropout(dropout_cls)
        self.fc = nn.Linear(hidden, classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.unsqueeze(1))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        out, _ = self.lstm(self.drop_in(x))
        past = out[:, :-1, :]
        current = out[:, -1, :]
        transformed = torch.tanh(self.attn_hidden(past))
        weights = torch.softmax(self.attn_score(transformed), dim=1)
        context = (weights * past).sum(dim=1)
        embedding = context + current
        return self.fc(self.drop_out(embedding))


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, *, no_null: bool) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if no_null:
        labels = sorted(v for v in np.unique(y_true).tolist() if int(v) != 0)
        mask = y_true != 0
        out["f1_macro_no_null"] = float(f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0))
        out["accuracy_no_null"] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return out


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    args,
    device: torch.device,
    *,
    no_null: bool,
    add_invalid_prefix: bool = False,
) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(x[start : start + args.batch_size]).to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = y
    if add_invalid_prefix and len(y_true):
        prefix = np.full(args.window - 1, y_true[0], dtype=y_true.dtype)
        y_true = np.concatenate([y_true, prefix])
        y_pred = np.concatenate([y_pred, prefix])
    return metric_bundle(y_true, y_pred, no_null=no_null)


def train_one(name: str, args, device: torch.device) -> dict:
    ref = PAPER_REFERENCES[name]
    x_train, y_train, x_val, y_val, x_test, y_test, meta = load_dataset(name, args)
    classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    model = AttentionDeepConvLSTM(
        channels=x_train.shape[2],
        classes=classes,
        dropout=args.dropout,
        dropout_rnn=args.dropout_rnn,
        dropout_cls=args.dropout_cls,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay) if args.lr_decay else None
    loss_fn = nn.CrossEntropyLoss()
    best_val = None
    best_state = None
    start_time = time.time()
    no_null = name in {"opportunity", "skoda"}
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
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, x_val, y_val, args, device, no_null=no_null)
            key = ref["paper_metric"]
            if best_val is None or val_metrics[key] > best_val[key]:
                best_val = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                "{} epoch={} loss={:.4f} val_{}={:.4f}".format(
                    name, epoch, float(np.mean(losses)), key, val_metrics[key]
                ),
                flush=True,
            )
        else:
            print(f"{name} epoch={epoch} loss={float(np.mean(losses)):.4f}", flush=True)
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        x_test,
        y_test,
        args,
        device,
        no_null=no_null,
        add_invalid_prefix=bool(args.sample_wise_test),
    )
    primary = ref["paper_metric"]
    gap = float(test_metrics[primary] - ref["paper_score"])
    runtime = time.time() - start_time
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": float(test_metrics[primary]),
        "metrics": test_metrics,
        "best_validation_metrics": best_val,
        "source": "attention_model_paper_dataset_reimplementation",
        "protocol_status": "paper_dataset_reimplementation",
        "paper_reference_metric": primary,
        "paper_reference_score": ref["paper_score"],
        "paper_reference_source": ref["source"],
        "paper_reference_note": ref["note"],
        "paper_reference_gap_signed": gap,
        "paper_reference_gap_abs": abs(gap),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range" if abs(gap) <= args.gap_threshold else "large_gap_check_protocol",
        "runtime_seconds": round(runtime, 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "optimizer": "RMSprop(lr={}, alpha=0.9)".format(args.lr),
        "lr_decay_per_epoch": args.lr_decay,
        "dropout": args.dropout,
        "dropout_rnn": args.dropout_rnn,
        "dropout_cls": args.dropout_cls,
        "device": str(device),
        "torch_version": torch.__version__,
        "window_size_samples": args.window,
        "stride_samples": args.stride,
        "sample_wise_test": bool(args.sample_wise_test),
        "train_windows": int(len(x_train)),
        "val_windows": int(len(x_val)),
        "test_windows": int(len(x_test)),
        "input_channels": int(x_train.shape[2]),
        "classes": classes,
        "reimplementation_note": "Paper Eq.3-5 attention on top of a DeepConvLSTM-style valid-convolution encoder; no official Attention Model author checkpoint was available.",
        "data_loader_meta": meta,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(index)
    return result


def write_entry(results: list[dict], args) -> Path:
    datasets = []
    for row in results:
        ref = next(v for v in PAPER_REFERENCES.values() if v["benchmark_id"] == row["benchmark_id"])
        datasets.append(
            {
                "benchmark_id": row["benchmark_id"],
                "dataset": ref["dataset"],
                "activity_task": "Original Attention Model paper dataset comparison.",
                "reported_effect": "Reimplementation {} {:.4f} vs paper {:.4f}; gap {:+.4f}.".format(
                    row["paper_reference_metric"],
                    row["primary_score"],
                    row["paper_reference_score"],
                    row["paper_reference_gap_signed"],
                ),
                "protocol_note": row["reimplementation_note"],
            }
        )
    entry = {
        "id": "attention_model",
        "method_name": "Attention Model",
        "year": 2018,
        "method_family": "CNN/LSTM + Attention",
        "submission_type": "platform",
        "coverage_status": "paper_dataset_reimplementation",
        "summary": "Attention Model paper-dataset reimplementation with direct gap comparison against the original paper table.",
        "implementation": "PyTorch DeepConvLSTM-style valid-convolution encoder with paper Eq.3-5 temporal attention; paper datasets/splits from the local Attend-And-Discriminate coverage repository.",
        "har_scope": "Wearable HAR on OPPORTUNITY, PAMAP2, and Skoda paper datasets.",
        "datasets": datasets,
        "results": results,
        "sources": ["https://arxiv.org/pdf/1805.07648"],
    }
    out_dir = RUNS_DIR / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


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
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_REFERENCES), default=sorted(PAPER_REFERENCES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dropout-rnn", type=float, default=0.25)
    parser.add_argument("--dropout-cls", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--stride-test", type=int, default=1)
    parser.add_argument("--sample-wise-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--force-mat", action="store_true")
    parser.add_argument("--run-tag", default="attention_model_paper_reimpl_20260725")
    args = parser.parse_args(argv)

    seed_all(args.seed)
    device = select_device(args)
    print(f"Using device: {device}", flush=True)
    results = []
    for name in args.datasets:
        result = train_one(name, args, device)
        results.append(result)
        print(
            "{} done: {}={:.4f}, paper={:.4f}, gap={:+.4f}, status={}".format(
                name,
                result["paper_reference_metric"],
                result["primary_score"],
                result["paper_reference_score"],
                result["paper_reference_gap_signed"],
                result["paper_reference_gap_status"],
            ),
            flush=True,
        )
    out_path = write_entry(results, args)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
