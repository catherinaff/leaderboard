"""Run paper-inspired raw-signal UCI-HAR neural baselines.

These are local canonical reruns inspired by common HAR architectures when the
original code is unavailable or not runnable in the current environment.
"""
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
RUN_DIR = ROOT / "experiments" / "runs" / "uci_har_paper_inspired"
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


class DCNN(nn.Module):
    def __init__(self, classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 64, 5, padding=2),
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


class BiLSTMS(nn.Module):
    def __init__(self, classes: int = 6) -> None:
        super().__init__()
        self.lstm = nn.LSTM(9, 128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(256, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class AttentionHAR(nn.Module):
    def __init__(self, classes: int = 6) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(9, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)
        self.fc = nn.Linear(256, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).transpose(1, 2)
        out, _ = self.gru(x)
        weights = torch.softmax(self.attn(out), dim=1)
        pooled = (weights * out).sum(dim=1)
        return self.fc(pooled)


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


def train_eval(model: nn.Module, name: str, x_train, y_train, x_test, y_test, args, device) -> dict:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = model.to(device)
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
        print(f"{name} epoch={epoch} loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test).to(device)).argmax(dim=1).cpu().numpy()

    acc = float(accuracy_score(y_test, pred))
    f1_macro = float(f1_score(y_test, pred, average="macro"))
    f1_weighted = float(f1_score(y_test, pred, average="weighted"))
    return {
        "primary_score": acc,
        "metrics": {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted},
        "runtime_seconds": round(time.time() - start, 3),
        "epochs": args.epochs,
        "seed": args.seed,
    }


def entry(method_id: str, name: str, family: str, result: dict) -> dict:
    return {
        "id": method_id,
        "method_name": name,
        "method_family": family,
        "submission_type": "platform",
        "coverage_status": "canonical_paper_inspired_rerun",
        "summary": f"{name} paper-inspired local implementation trained on canonical UCI-HAR raw inertial signals.",
        "implementation": "PyTorch local implementation using the paper family architecture rather than an official author checkpoint.",
        "har_scope": "Raw-signal smartphone HAR on UCI-HAR fixed split.",
        "datasets": [
            {
                "benchmark_id": "uci_har",
                "dataset": "UCI-HAR",
                "activity_task": "6-class activity recognition from raw 9-channel inertial windows.",
                "subjects": "30 total; official fixed train subjects 1-21 and test subjects 22-30.",
                "device_placement": "Waist-mounted Samsung Galaxy S II smartphone.",
                "reported_effect": f"Local canonical paper-inspired rerun accuracy {result['metrics']['accuracy']:.4f}, f1_macro {result['metrics']['f1_macro']:.4f}.",
                "protocol_note": "Canonical split, local implementation. Marked paper-inspired because the original code/checkpoint was not used."
            }
        ],
        "results": [
            {
                "benchmark_id": "uci_har",
                "primary_score": result["primary_score"],
                "metrics": result["metrics"],
                "source": "local_paper_inspired_rerun",
                "protocol_status": "canonical",
                "runtime_seconds": result["runtime_seconds"],
                "epochs": result["epochs"],
                "seed": result["seed"]
            }
        ],
        "sources": [
            "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    seed_all(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x_train, y_train = load_raw_split("train")
    x_test, y_test = load_raw_split("test")
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    specs = [
        ("dcnn", "DCNN", "Deep CNN", DCNN()),
        ("blstm_s", "b-LSTM-S", "Bidirectional LSTM", BiLSTMS()),
        ("attention_model", "Attention Model", "CNN/GRU + Attention", AttentionHAR()),
    ]
    outputs = []
    for method_id, name, family, model in specs:
        result = train_eval(model, name, x_train, y_train, x_test, y_test, args, device)
        outputs.append(entry(method_id, name, family, result))
        print(f"{name}: accuracy={result['metrics']['accuracy']:.4f}, f1_macro={result['metrics']['f1_macro']:.4f}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = RUN_DIR / "results.json"
    out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
