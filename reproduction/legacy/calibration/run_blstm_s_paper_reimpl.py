"""Run b-LSTM-S on A&D paper-comparison datasets.

The leaderboard currently only has paper table values for b-LSTM-S on
OPPORTUNITY and PAMAP2. This runner uses the same local
Attend-And-Discriminate paper-coverage dataset files that we used for recent
Attention Model calibration, then compares a stacked bidirectional LSTM rerun
against the cited table values.
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
        "paper_score": 0.684,
        "paper_metric": "f1_macro",
        "source": "Attend And Discriminate 2020 Table 2, b-LSTM-S row; cited from Hammerla et al. 2016.",
        "note": "A&D Table 2 reports mean F1 68.4 for b-LSTM-S on OPPORTUNITY.",
    },
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "classes": 12,
        "input_dim": 52,
        "paper_score": 0.838,
        "paper_metric": "f1_macro",
        "source": "Attend And Discriminate 2020 Table 2, b-LSTM-S row; cited from Hammerla et al. 2016.",
        "note": "A&D Table 2 reports mean F1 83.8 for b-LSTM-S on PAMAP2.",
    },
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
    )


def load_from_processed(name: str, split: str, *, sample_wise_test: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    filename = "test_sample_wise.npz" if split == "test" and sample_wise_test else f"{split}.npz"
    path = AD_REPO / "data" / name / "processed" / "24_12" / filename
    data = np.load(path)
    return data["data"].astype(np.float32), data["target"].astype(np.int64), {"loader": "processed_npz_24_12", "path": str(path)}


def load_from_mat(name: str, window: int, stride: int, stride_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    path = AD_REPO / "dataset" / f"{name}.mat"
    contents = sio.loadmat(path)
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
    return x_train, y_train, x_val, y_val, x_test, y_test, {
        "loader": "mat_rewindowed",
        "mat_path": str(path),
        "window_size_samples": window,
        "train_stride_samples": stride,
        "test_stride_samples": stride_test,
    }


def load_dataset(name: str, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    # The local OPPORTUNITY processed train archive on 245 has a CRC error, so
    # regenerate that split from the .mat file. PAMAP2 processed files are OK.
    if name == "pamap2" and not args.force_mat:
        x_train, y_train, train_meta = load_from_processed(name, "train", sample_wise_test=False)
        x_val, y_val, val_meta = load_from_processed(name, "val", sample_wise_test=False)
        x_test, y_test, test_meta = load_from_processed(name, "test", sample_wise_test=args.sample_wise_test)
        return x_train, y_train, x_val, y_val, x_test, y_test, {
            "source_repo": str(AD_REPO),
            "split_source": "Attend-And-Discriminate paper-coverage processed files",
            "train": train_meta,
            "val": val_meta,
            "test": test_meta,
        }
    stride_test = args.stride_test if args.sample_wise_test else args.stride
    x_train, y_train, x_val, y_val, x_test, y_test, meta = load_from_mat(name, args.window, args.stride, stride_test)
    meta["source_repo"] = str(AD_REPO)
    meta["split_source"] = "Attend-And-Discriminate paper-coverage .mat file"
    return x_train, y_train, x_val, y_val, x_test, y_test, meta


class BLSTMS(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int, layers: int, dropout: float, dropout_cls: float) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            channels,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(dropout_cls)
        self.fc = nn.Linear(hidden * 2, classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.input_dropout(x))
        # Last layer forward and backward states.
        feature = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.output_dropout(feature))


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, args, device: torch.device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(x[start : start + args.batch_size]).to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = y
    if args.sample_wise_test and len(y_true):
        prefix = np.full(args.window - 1, y_true[0], dtype=y_true.dtype)
        y_true = np.concatenate([y_true, prefix])
        y_pred = np.concatenate([y_pred, prefix])
    return metrics_for(y_true, y_pred)


def train_one(name: str, args, device: torch.device) -> dict:
    ref = PAPER_REFERENCES[name]
    x_train, y_train, x_val, y_val, x_test, y_test, meta = load_dataset(name, args)
    classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    model = BLSTMS(
        channels=x_train.shape[2],
        classes=classes,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        dropout_cls=args.dropout_cls,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_decay) if args.lr_step > 0 else None
    loss_fn = nn.CrossEntropyLoss()
    best_val = None
    best_state = None
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, x_val, y_val, args, device)
            if best_val is None or val_metrics[ref["paper_metric"]] > best_val[ref["paper_metric"]]:
                best_val = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                "{} epoch={} loss={:.4f} val_{}={:.4f}".format(
                    name, epoch, float(np.mean(losses)), ref["paper_metric"], val_metrics[ref["paper_metric"]]
                ),
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, x_test, y_test, args, device)
    primary = ref["paper_metric"]
    gap = float(test_metrics[primary] - ref["paper_score"])
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": float(test_metrics[primary]),
        "primary_metric": primary,
        "metrics": test_metrics,
        "best_validation_metrics": best_val,
        "source": "blstm_s_paper_dataset_reimplementation",
        "protocol_status": "paper_dataset_reimplementation",
        "paper_reimplementation": True,
        "paper_reference_metric": primary,
        "paper_reference_score": ref["paper_score"],
        "paper_reference_source": ref["source"],
        "paper_reference_note": ref["note"],
        "paper_reference_gap_signed": gap,
        "paper_reference_gap_abs": abs(gap),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range" if abs(gap) <= args.gap_threshold else "large_gap_check_protocol",
        "calibration_status": "passed" if abs(gap) <= args.gap_threshold else "failed",
        "runtime_seconds": round(time.time() - start_time, 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "optimizer": "Adam(lr={})".format(args.lr),
        "lr_decay": args.lr_decay,
        "lr_step": args.lr_step,
        "hidden_units": args.hidden,
        "lstm_layers": args.layers,
        "bidirectional": True,
        "dropout": args.dropout,
        "dropout_cls": args.dropout_cls,
        "clip_grad": args.clip_grad,
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
        "reimplementation_note": "Stacked bidirectional LSTM classifier over A&D paper-comparison windows; original b-LSTM-S checkpoint/code was not available in this workspace.",
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
                "activity_task": "Original b-LSTM-S paper comparison dataset.",
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
        "id": "blstm_s",
        "method_name": "b-LSTM-S",
        "year": 2016,
        "method_family": "Bidirectional LSTM",
        "submission_type": "platform",
        "paper_url": "https://www.ijcai.org/Proceedings/16/Papers/220.pdf",
        "coverage_status": "paper_dataset_reimplementation",
        "summary": "b-LSTM-S paper-dataset reimplementation with direct gap comparison against the A&D Table 2 cited values.",
        "implementation": "PyTorch stacked bidirectional LSTM trained on local Attend-And-Discriminate paper-comparison dataset files.",
        "har_scope": "Wearable HAR paper-comparison rows for OPPORTUNITY and PAMAP2.",
        "datasets": datasets,
        "results": results,
        "sources": [
            "https://www.ijcai.org/Proceedings/16/Papers/220.pdf",
            "https://arxiv.org/abs/2007.07172",
        ],
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--lr-step", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dropout-cls", type=float, default=0.5)
    parser.add_argument("--clip-grad", type=float, default=0.0)
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
    parser.add_argument("--run-tag", default="blstm_s_paper_reimpl_20260725")
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
