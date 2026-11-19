"""Run b-LSTM-S with the IJCAI 2016 sample-level protocol.

Hammerla et al. train LSTM-S / b-LSTM-S by feeding raw movement samples in the
order they were recorded and predicting a label for each sample. This differs
from the leaderboard's earlier paper-dataset runner, which classified a window
from its final timestep label.
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


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
AD_REPO = ROOT / "experiments" / "repos" / "Attend-And-Discriminate"

PAPER_REFERENCES = {
    "opportunity": {
        "benchmark_id": "opportunity_a",
        "dataset": "OPPORTUNITY",
        "activity_task": "18 mid-level gesture annotations, 79-D sensor subset.",
        "paper_metrics": {"f1_macro": 0.745, "f1_weighted": 0.927},
        "paper_note": "Paper reports OPPORTUNITY Fm 0.745 and Fw 0.927.",
        "split_note": "Subject/run split follows Hammerla et al.: subject 1 run 2 validation; subject 2/3 runs 4/5 test; remaining data train.",
    },
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "activity_task": "12 activity recognition using the Hammerla et al. sample-level split.",
        "paper_metrics": {"f1_macro": 0.868},
        "paper_note": "Paper reports PAMAP2 Fm 0.868.",
        "split_note": "Subject/run split follows Hammerla et al.: subject 5 runs 1/2 validation; subject 6 runs 1/2 test; remaining data train.",
    },
}
GAP_THRESHOLD = 0.05


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_splits(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
    )


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    path = AD_REPO / "dataset" / f"{name}.mat"
    mat = sio.loadmat(path)
    if name == "opportunity":
        x_train = mat["trainingData"].astype(np.float32).T
        y_train = mat["trainingLabels"].reshape(-1).astype(np.int64) - 1
        x_val = mat["valData"].astype(np.float32).T
        y_val = mat["valLabels"].reshape(-1).astype(np.int64) - 1
        x_test = mat["testingData"].astype(np.float32).T
        y_test = mat["testingLabels"].reshape(-1).astype(np.int64) - 1
    elif name == "pamap2":
        x_train = mat["X_train"].astype(np.float32)
        y_train = mat["y_train"].reshape(-1).astype(np.int64)
        x_val = mat["X_valid"].astype(np.float32)
        y_val = mat["y_valid"].reshape(-1).astype(np.int64)
        x_test = mat["X_test"].astype(np.float32)
        y_test = mat["y_test"].reshape(-1).astype(np.int64)
    else:
        raise KeyError(name)

    x_train, x_val, x_test = normalize_splits(x_train, x_val, x_test)
    meta = {
        "mat_path": str(path),
        "feature_dim": int(x_train.shape[1]),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "split_note": PAPER_REFERENCES[name]["split_note"],
        "normalization": "z-score using training samples only",
    }
    return x_train, y_train, x_val, y_val, x_test, y_test, meta


class BLSTMSequenceTagger(nn.Module):
    def __init__(self, input_dim: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
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
        out, _ = self.lstm(self.input_dropout(x))
        return self.fc(out)


def sequence_batches(n: int, batch_size: int, seq_len: int, *, rng: np.random.Generator):
    positions = rng.integers(0, max(1, n - seq_len), size=batch_size)
    steps = max(1, n // (batch_size * seq_len))
    for _ in range(steps):
        yield positions.copy()
        positions = (positions + seq_len) % max(1, n - seq_len)


def apply_max_norm(model: nn.Module, max_norm: float) -> None:
    if max_norm <= 0:
        return
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim < 2:
                continue
            norm = param.norm(2, dim=0, keepdim=True).clamp_min(1e-8)
            desired = torch.clamp(norm, max=max_norm)
            param.mul_(desired / norm)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, seq_len: int, device: torch.device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), seq_len):
            xb = torch.from_numpy(x[start : start + seq_len]).unsqueeze(0).to(device)
            pred = model(xb).argmax(dim=-1).squeeze(0).cpu().numpy()
            preds.append(pred)
    return metrics_for(y, np.concatenate(preds)[: len(y)])


def train(args) -> tuple[dict, Path]:
    seed_all(args.seed)
    device = select_device(args)
    ref = PAPER_REFERENCES[args.dataset]
    paper_fm = ref["paper_metrics"]["f1_macro"]
    x_train, y_train, x_val, y_val, x_test, y_test, meta = load_dataset(args.dataset)
    classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    model = BLSTMSequenceTagger(x_train.shape[1], args.hidden, classes, args.dropout).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)
    best_val = None
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for positions in sequence_batches(len(x_train), args.batch_size, args.seq_len, rng=rng):
            xb = np.stack([x_train[p : p + args.seq_len] for p in positions])
            yb = np.stack([y_train[p : p + args.seq_len] for p in positions])
            xb_t = torch.from_numpy(xb).to(device, non_blocking=True)
            yb_t = torch.from_numpy(yb).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_t)
            loss = loss_fn(logits.reshape(-1, classes), yb_t.reshape(-1))
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            apply_max_norm(model, args.max_norm)
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, x_val, y_val, args.seq_len, device)
        score = val_metrics["f1_macro"]
        improved = best_val is None or score > best_val["f1_macro"]
        if improved:
            best_val = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            "{} epoch={} loss={:.4f} val_f1_macro={:.4f} best={:.4f}".format(
                args.dataset, epoch, float(np.mean(losses)), score, best_val["f1_macro"]
            ),
            flush=True,
        )
        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, x_test, y_test, args.seq_len, device)
    runtime = time.time() - start_time
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": test_metrics["f1_macro"],
        "primary_metric": "f1_macro",
        "metrics": test_metrics,
        "best_validation_metrics": best_val,
        "source": "blstm_s_hammerla_sample_level_reimplementation",
        "protocol_status": "paper_dataset_reimplementation",
        "paper_reimplementation": True,
        "paper_reference_metric": "f1_macro",
        "paper_reference_score": paper_fm,
        "paper_reference_metrics": ref["paper_metrics"],
        "paper_reference_source": f"Hammerla, Halloran & Ploetz IJCAI 2016 Table 2, b-LSTM-S {ref['dataset']} row.",
        "paper_reference_note": ref["paper_note"],
        "paper_reference_gap_signed": float(test_metrics["f1_macro"] - paper_fm),
        "paper_reference_gap_abs": float(abs(test_metrics["f1_macro"] - paper_fm)),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range"
        if abs(test_metrics["f1_macro"] - paper_fm) <= args.gap_threshold
        else "large_gap_check_protocol",
        "calibration_status": "passed"
        if abs(test_metrics["f1_macro"] - paper_fm) <= args.gap_threshold
        else "failed",
        "runtime_seconds": round(runtime, 3),
        "epochs_run": epoch,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length_samples": args.seq_len,
        "optimizer": "Adagrad(lr={}, weight_decay={})".format(args.lr, args.weight_decay),
        "hidden_units": args.hidden,
        "lstm_layers": 1,
        "bidirectional": True,
        "dropout": args.dropout,
        "max_norm": args.max_norm,
        "clip_grad": args.clip_grad,
        "device": str(device),
        "torch_version": torch.__version__,
        "classes": classes,
        "input_channels": int(x_train.shape[1]),
        "reimplementation_note": "Sample-level b-LSTM-S: every timestep predicts a label, using ordered random-start sequences as described in IJCAI 2016 Section 3.4.",
        "data_loader_meta": meta,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(index)
    entry = {
        "id": "blstm_s",
        "method_name": "b-LSTM-S",
        "year": 2016,
        "method_family": "Bidirectional LSTM",
        "submission_type": "platform",
        "paper_url": "https://www.ijcai.org/Proceedings/16/Papers/220.pdf",
        "coverage_status": "paper_dataset_reimplementation",
        "summary": f"b-LSTM-S sample-level {ref['dataset']} reimplementation following Hammerla et al. IJCAI 2016.",
        "implementation": "PyTorch one-layer bidirectional LSTM sequence tagger trained sample-by-sample with Adagrad and max-norm regularization.",
        "har_scope": "Sample-level wearable HAR following Hammerla et al. IJCAI 2016.",
        "datasets": [
            {
                "benchmark_id": ref["benchmark_id"],
                "dataset": ref["dataset"],
                "activity_task": ref["activity_task"],
                "reported_effect": "Reimplementation f1_macro {:.4f} vs paper {:.4f}; gap {:+.4f}.".format(
                    result["primary_score"], paper_fm, result["paper_reference_gap_signed"]
                ),
                "protocol_note": ref["split_note"],
            }
        ],
        "results": [result],
        "sources": ["https://www.ijcai.org/Proceedings/16/Papers/220.pdf"],
    }
    out_dir = RUNS_DIR / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result, out_path


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
    parser.add_argument("--dataset", choices=sorted(PAPER_REFERENCES), default="opportunity")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-norm", type=float, default=4.0)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args(argv)

    if args.run_tag is None:
        args.run_tag = f"blstm_s_{args.dataset}_hammerla_sample_20260725"
    result, out_path = train(args)
    print(
        "{} b-LSTM-S sample-level: f1_macro={:.6f}, paper={:.6f}, gap={:+.6f}, status={}, wrote {}".format(
            args.dataset,
            result["primary_score"],
            result["paper_reference_score"],
            result["paper_reference_gap_signed"],
            result["paper_reference_gap_status"],
            out_path,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
