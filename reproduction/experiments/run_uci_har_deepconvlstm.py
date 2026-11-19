"""Run a DeepConvLSTM-style canonical UCI-HAR model on raw inertial signals."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "datasets" / "uci_har" / "UCI HAR Dataset"
RUN_DIR = ROOT / "experiments" / "runs" / "uci_har_deepconvlstm"
SIGNALS = [
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


class DeepConvLSTM(nn.Module):
    def __init__(self, channels: int = 9, classes: int = 6) -> None:
        super().__init__()
        layers = []
        in_ch = channels
        for _ in range(4):
            layers.extend([nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.ReLU()])
            in_ch = 64
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_raw_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    sig_dir = DATA_DIR / split / "Inertial Signals"
    arrays = [np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32) for name in SIGNALS]
    x = np.stack(arrays, axis=1)
    y = np.loadtxt(DATA_DIR / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not (DATA_DIR / "train" / "Inertial Signals").is_dir():
        raise SystemExit("UCI-HAR data missing. Run experiments/download_uci_har.py first.")

    seed_all(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    x_train, y_train = load_raw_split("train")
    x_test, y_test = load_raw_split("test")
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_x = torch.from_numpy(x_test).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = DeepConvLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch} loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(test_x).argmax(dim=1).cpu().numpy()

    acc = float(accuracy_score(y_test, pred))
    f1_macro = float(f1_score(y_test, pred, average="macro"))
    f1_weighted = float(f1_score(y_test, pred, average="weighted"))
    runtime = round(time.time() - start, 3)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUN_DIR / "model.pt")
    entry = {
        "id": "deepconvlstm",
        "method_name": "DeepConvLSTM",
        "submission_type": "platform",
        "coverage_status": "canonical_partial_rerun",
        "datasets": [
            {
                "benchmark_id": "uci_har",
                "dataset": "UCI-HAR",
                "activity_task": "6-class activity recognition from raw 9-channel inertial windows.",
                "subjects": "30 total; official fixed train subjects 1-21 and test subjects 22-30.",
                "device_placement": "Waist-mounted Samsung Galaxy S II smartphone.",
                "reported_effect": f"Local canonical rerun accuracy {acc:.4f}, f1_macro {f1_macro:.4f}.",
                "protocol_note": f"DeepConvLSTM-style implementation run locally for {args.epochs} epochs on raw UCI-HAR inertial signals."
            }
        ],
        "results": [
            {
                "benchmark_id": "uci_har",
                "primary_score": acc,
                "metrics": {
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted
                },
                "source": "local_deepconvlstm_style_rerun",
                "protocol_status": "canonical",
                "runtime_seconds": runtime,
                "epochs": args.epochs,
                "seed": args.seed
            }
        ],
        "sources": [
            "https://www.mdpi.com/1424-8220/16/1/115",
            "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
        ]
    }
    out = RUN_DIR / "results.json"
    out.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"DeepConvLSTM-style: accuracy={acc:.4f}, f1_macro={f1_macro:.4f}, runtime={runtime}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
