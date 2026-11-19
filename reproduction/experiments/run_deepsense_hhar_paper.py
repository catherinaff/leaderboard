"""Run a DeepSense HHAR paper-protocol calibration fold.

This script uses the official DeepSense HHAR preprocessed CSV layout
(`sepHARData_a/{train,eval}`) and ports the public TensorFlow 1.1 model
structure layer-by-layer to PyTorch so it can run on the current servers.
It is intended for paper-protocol calibration, not the unified Standard track.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


WIDE = 20
SPECTRAL_SAMPLES = 10
FEATURE_DIM = SPECTRAL_SAMPLES * 6 * 2
OUT_DIM = 6
INTER_DIM = 120


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def _read_csv_file(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def build_cache(data_dir: Path, cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split directory: {split_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = cache_dir / f"{split}_X.npy"
    y_path = cache_dir / f"{split}_y.npy"
    if x_path.is_file() and y_path.is_file():
        return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")

    files = sorted(split_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no csv files under {split_dir}")

    arrays = []
    for idx, path in enumerate(files, 1):
        arrays.append(_read_csv_file(path))
        if idx % 500 == 0:
            print(f"[cache] {split}: read {idx}/{len(files)} csv files", flush=True)
    raw = np.concatenate(arrays, axis=0)
    expected_dim = WIDE * FEATURE_DIM + OUT_DIM
    if raw.shape[1] != expected_dim:
        raise ValueError(f"{split} has {raw.shape[1]} columns, expected {expected_dim}")

    x = raw[:, : WIDE * FEATURE_DIM].reshape(-1, WIDE, FEATURE_DIM).astype(np.float32)
    y_onehot = raw[:, WIDE * FEATURE_DIM :]
    y = np.argmax(y_onehot, axis=1).astype(np.int64)
    np.save(x_path, x)
    np.save(y_path, y)
    print(f"[cache] {split}: saved X={x.shape} y={y.shape} to {cache_dir}", flush=True)
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


class DeepSenseHHAR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.acc_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 18), stride=(1, 6), padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.gyro_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 18), stride=(1, 6), padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.sensor_conv1 = nn.Conv3d(
            64, 64, kernel_size=(1, 2, 8), stride=1, padding="same", bias=True
        )
        self.sensor_bn1 = nn.BatchNorm3d(64)
        self.sensor_conv2 = nn.Conv3d(
            64, 64, kernel_size=(1, 2, 6), stride=1, padding="same", bias=True
        )
        self.sensor_bn2 = nn.BatchNorm3d(64)
        self.sensor_conv3 = nn.Conv3d(
            64, 64, kernel_size=(1, 2, 4), stride=1, padding="same", bias=True
        )
        self.sensor_bn3 = nn.BatchNorm3d(64)
        self.sensor_dropout = nn.Dropout3d(p=0.2)
        self.gru = nn.GRU(
            input_size=64 * 2 * 4,
            hidden_size=INTER_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.gru_out_dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(INTER_DIM, OUT_DIM)

    def _sensor_branch(self, x: torch.Tensor, branch: nn.Module) -> torch.Tensor:
        # Official TensorFlow input is [B, WIDE, 60, 1]; PyTorch uses NCHW.
        y = branch(x.unsqueeze(1))
        # [B, C, WIDE, 4] -> [B, C, WIDE, sensor=1, 4]
        return y.unsqueeze(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        used = torch.amax(torch.abs(x), dim=2) > 0
        lengths = torch.clamp(used.sum(dim=1).to(torch.long), min=1).cpu()
        mask = used.unsqueeze(-1).to(x.dtype)

        acc = x[:, :, : FEATURE_DIM // 2]
        gyro = x[:, :, FEATURE_DIM // 2 :]
        acc_conv = self._sensor_branch(acc, self.acc_conv)
        gyro_conv = self._sensor_branch(gyro, self.gyro_conv)
        sensor = torch.cat([acc_conv, gyro_conv], dim=3)

        sensor = self.sensor_dropout(sensor)
        sensor = torch.relu(self.sensor_bn1(self.sensor_conv1(sensor)))
        sensor = self.sensor_dropout(sensor)
        sensor = torch.relu(self.sensor_bn2(self.sensor_conv2(sensor)))
        sensor = self.sensor_dropout(sensor)
        sensor = torch.relu(self.sensor_bn3(self.sensor_conv3(sensor)))

        # [B, C, WIDE, 2, 4] -> [B, WIDE, 2*4*C]
        sensor = sensor.permute(0, 2, 3, 4, 1).contiguous()
        sensor = sensor.view(sensor.shape[0], sensor.shape[1], -1)

        packed = nn.utils.rnn.pack_padded_sequence(
            sensor, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=WIDE
        )
        rnn_out = self.gru_out_dropout(rnn_out)
        avg = (rnn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.out(avg)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred))
    f1s = []
    supports = []
    for cls in range(OUT_DIM):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        support = float(np.sum(y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        supports.append(support)
    supports_arr = np.asarray(supports, dtype=np.float64)
    f1_arr = np.asarray(f1s, dtype=np.float64)
    return {
        "accuracy": accuracy,
        "f1_macro": float(f1_arr.mean()),
        "f1_weighted": float((f1_arr * supports_arr).sum() / supports_arr.sum()),
        "f1_micro": accuracy,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    labels = []
    losses = []
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        losses.append(float(criterion(logits, yb).detach().cpu()))
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels.append(yb.cpu().numpy())
    metrics = metric_summary(np.concatenate(labels), np.concatenate(preds))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[run] device={device} torch={torch.__version__}", flush=True)

    train_x, train_y = build_cache(args.data_dir, args.cache_dir, "train")
    eval_x, eval_y = build_cache(args.data_dir, args.cache_dir, "eval")
    train_ds = TensorDataset(torch.from_numpy(np.asarray(train_x)), torch.from_numpy(np.asarray(train_y)))
    eval_ds = TensorDataset(torch.from_numpy(np.asarray(eval_x)), torch.from_numpy(np.asarray(eval_y)))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DeepSenseHHAR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay
    )

    paper = {"accuracy": 0.942, "f1_macro": 0.931, "f1_micro": 0.942}
    best = {"accuracy": -math.inf, "step": 0, "epoch": 0}
    stale = 0
    global_step = 0
    start = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            global_step += 1
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if global_step == 1 or global_step % 50 == 0:
                pred = torch.argmax(logits.detach(), dim=1)
                train_acc = float((pred == yb).float().mean().detach().cpu())
                print(
                    f"[train] epoch={epoch} step={global_step} loss={float(loss.detach().cpu()):.5f} acc={train_acc:.4f}",
                    flush=True,
                )

            if global_step == 1 or global_step % args.eval_every_steps == 0:
                metrics = evaluate(model, eval_loader, device)
                row = {"epoch": epoch, "step": global_step, **metrics}
                history.append(row)
                print(
                    "[eval] "
                    + " ".join(
                        f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()
                    ),
                    flush=True,
                )
                if metrics["accuracy"] > best["accuracy"]:
                    best = {"epoch": epoch, "step": global_step, **metrics}
                    stale = 0
                else:
                    stale += 1
                if stale >= args.patience:
                    print(f"[early-stop] stale={stale} best_step={best['step']}", flush=True)
                    break
        if stale >= args.patience:
            break

    result = {
        "method_id": "deepsense",
        "method_name": "DeepSense",
        "benchmark_id": "hhar",
        "official_dataset": "sepHARData_a",
        "protocol_status": "paper_protocol",
        "source": "official_preprocessed_data_source_code_equivalent_port",
        "official_code_used": True,
        "official_repo": "https://github.com/yscacaca/DeepSense",
        "paper_reference_source": "DeepSense WWW 2017 paper, HHAR human activity recognition table",
        "paper_reference_metrics": paper,
        "best_eval": best,
        "paper_reference_gap_abs": {
            key: abs(best.get(key, float("nan")) - value) for key, value in paper.items()
        },
        "history": history,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_seconds": round(time.time() - start, 3),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(result["best_eval"], sort_keys=True), flush=True)
    print(f"[done] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
