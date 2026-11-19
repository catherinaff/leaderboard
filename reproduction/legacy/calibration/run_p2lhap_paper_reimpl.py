"""Paper-style P2LHAP reproduction.

The P2LHAP paper does not appear to provide public author code. This runner
implements the core paper ingredients for HAR classification: non-overlapping
sensor patches, a channel-independent Transformer encoder, a Transformer
decoder that predicts patch labels, weighted patch-level CE, and truncated MSE
label-smoothing loss. It reports the gap against the paper's Table I values.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "experiments" / "datasets"
RUNS_DIR = ROOT / "experiments" / "runs"
ADAMASK_DATASETS = ROOT / "experiments" / "repos" / "AdaMaskNet" / "dataset"
GAP_THRESHOLD = 0.05

PAMAP2_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PAMAP2_LABEL = {activity: i for i, activity in enumerate(PAMAP2_ACTIVITIES)}

PAPER_REFERENCES = {
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "paper_metrics": {"accuracy": 0.9871, "f1_weighted": 0.9892},
        "primary_metric": "accuracy",
        "source": "P2LHAP paper Table I, PAMAP2 row: Acc/F1 98.71/98.92%.",
        "protocol_note": "Paper states continuous dataset split 0.7/0.1/0.2, batch 64, Adam lr 1e-4, and recognition patch size 10.",
    },
    "wisdm": {
        "benchmark_id": "wisdm",
        "dataset": "WISDM",
        "paper_metrics": {"accuracy": 0.9751, "f1_weighted": 0.9752},
        "primary_metric": "accuracy",
        "source": "P2LHAP paper Table I, WISDM row: Acc/F1 97.51/97.52%.",
        "protocol_note": "Paper states continuous dataset split 0.7/0.1/0.2, batch 64, Adam lr 1e-4, and recognition patch size 10.",
    },
    "unimib_shar": {
        "benchmark_id": "unimib_shar",
        "dataset": "UniMiB-SHAR",
        "paper_metrics": {"accuracy": 0.8151, "f1_weighted": 0.8204},
        "primary_metric": "accuracy",
        "source": "P2LHAP paper Table I, UniMiB-SHAR row: Acc/F1 81.51/82.04%.",
        "protocol_note": "Paper states continuous dataset split 0.7/0.1/0.2, batch 64, Adam lr 1e-4, and recognition patch size 10.",
    },
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def fill_missing(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    means = np.nanmean(np.where(np.isfinite(out), out, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.take(means, np.where(bad)[1])
    return out


def normalize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


def ensure_time_channel(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected (N,T,C) or (N,C,T), got {x.shape}")
    if x.shape[1] < x.shape[2]:
        x = np.transpose(x, (0, 2, 1))
    return x.astype(np.float32)


def load_adamask_split(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    folder = ADAMASK_DATASETS / name
    x_train = ensure_time_channel(np.load(folder / "x_train.npy"))
    y_train = np.load(folder / "y_train.npy").astype(np.int64)
    x_test = ensure_time_channel(np.load(folder / "x_test.npy"))
    y_test = np.load(folder / "y_test.npy").astype(np.int64)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    y = y - y.min()
    return x, y, {
        "source": str(folder),
        "original_split": {"train": int(len(x_train)), "test": int(len(x_test))},
        "note": "Existing public split arrays are recombined and re-split 0.7/0.1/0.2 to match the P2LHAP paper-level protocol.",
    }


def pamap2_channel_indices() -> list[int]:
    cols: list[int] = []
    for base in [3, 20, 37]:
        cols.extend(range(base + 1, base + 4))
        cols.extend(range(base + 7, base + 13))
    return cols


def sliding_windows(values: np.ndarray, labels: np.ndarray, window: int, stride: int) -> tuple[list[np.ndarray], list[int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, len(values) - window + 1, stride):
        end = start + window
        chunk = labels[start:end].astype(np.int64)
        counts = np.bincount(chunk)
        xs.append(values[start:end].astype(np.float32))
        ys.append(int(counts.argmax()))
    return xs, ys


def load_pamap2(args) -> tuple[np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "pamap2" / "PAMAP2_Dataset" / "Protocol"
    if not data_dir.is_dir():
        raise SystemExit(f"PAMAP2 raw directory not found: {data_dir}")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    cols = pamap2_channel_indices()
    files = []
    for path in sorted(data_dir.glob("subject*.dat")):
        raw = np.loadtxt(path, dtype=np.float32)
        activity = raw[:, 1].astype(int)
        values = fill_missing(raw[:, cols])
        start = 0
        file_windows = 0
        while start < len(values):
            act = int(activity[start])
            end = start + 1
            while end < len(values) and int(activity[end]) == act:
                end += 1
            if act in PAMAP2_LABEL:
                labels = np.full(end - start, PAMAP2_LABEL[act], dtype=np.int64)
                part_x, part_y = sliding_windows(values[start:end], labels, args.window_size, args.window_stride)
                xs.extend(part_x)
                ys.extend(part_y)
                file_windows += len(part_y)
            start = end
        files.append({"file": path.name, "windows": file_windows})
    if not xs:
        raise SystemExit("No PAMAP2 windows generated.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), {
        "source": str(data_dir),
        "files": files,
        "channels": "27 IMU channels: acceleration, gyroscope, and magnetometer from hand/chest/ankle; optional activities excluded.",
        "window_size": args.window_size,
        "window_stride": args.window_stride,
    }


def split_70_10_20(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    train_x, tmp_x, train_y, tmp_y = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    val_x, test_x, val_y, test_y = train_test_split(
        tmp_x, tmp_y, test_size=2.0 / 3.0, random_state=seed, stratify=tmp_y
    )
    return train_x, train_y, val_x, val_y, test_x, test_y


def load_dataset(name: str, args) -> tuple[np.ndarray, ...]:
    if name == "wisdm":
        x, y, meta = load_adamask_split("WISDM")
    elif name == "unimib_shar":
        x, y, meta = load_adamask_split("UniMiB_SHAR")
    elif name == "pamap2":
        x, y, meta = load_pamap2(args)
    else:
        raise KeyError(name)
    train_x, train_y, val_x, val_y, test_x, test_y = split_70_10_20(x, y, args.seed)
    train_x, val_x, test_x = normalize_by_train(train_x, val_x, test_x)
    meta.update({
        "split": "stratified random 0.7/0.1/0.2",
        "samples": {"train": int(len(train_y)), "val": int(len(val_y)), "test": int(len(test_y))},
        "shape": {"time_steps": int(train_x.shape[1]), "channels": int(train_x.shape[2]), "classes": int(y.max() + 1)},
    })
    return train_x, train_y, val_x, val_y, test_x, test_y, meta


class P2LHAPNet(nn.Module):
    def __init__(
        self,
        time_steps: int,
        channels: int,
        classes: int,
        patch_size: int,
        d_model: int,
        heads: int,
        enc_layers: int,
        dec_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if time_steps < patch_size:
            raise ValueError(f"time_steps={time_steps} must be >= patch_size={patch_size}")
        self.patch_size = patch_size
        self.patches = time_steps // patch_size
        self.trimmed_steps = self.patches * patch_size
        self.channels = channels
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.patches, d_model))
        self.query = nn.Parameter(torch.zeros(1, self.patches, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        dec = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=dec_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.trimmed_steps, :]
        bsz = x.shape[0]
        patches = x.transpose(1, 2).reshape(bsz * self.channels, self.patches, self.patch_size)
        tokens = self.patch_embed(patches) + self.pos
        memory = self.encoder(tokens)
        memory = memory.reshape(bsz, self.channels, self.patches, -1).mean(dim=1)
        query = self.query.expand(bsz, -1, -1)
        decoded = self.decoder(query, memory)
        return self.classifier(self.norm(decoded))


def class_weights(y: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    return torch.from_numpy(counts.max() / counts)


def tmse_loss(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    diff = (logp[:, 1:, :] - logp[:, :-1, :]).abs()
    return torch.clamp(diff, max=threshold).pow(2).mean()


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def metrics_for(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def evaluate(model: P2LHAPNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            pred = logits.mean(dim=1).argmax(dim=1).cpu().numpy()
            preds.append(pred)
            labels.append(yb.numpy())
    return metrics_for(np.concatenate(labels), np.concatenate(preds))


def train_one_dataset(name: str, args, device: torch.device) -> dict:
    ref = PAPER_REFERENCES[name]
    train_x, train_y, val_x, val_y, test_x, test_y, meta = load_dataset(name, args)
    classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)
    model = P2LHAPNet(
        time_steps=train_x.shape[1],
        channels=train_x.shape[2],
        classes=classes,
        patch_size=args.patch_size,
        d_model=args.d_model,
        heads=args.heads,
        enc_layers=args.encoder_layers,
        dec_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    weights = class_weights(train_y, classes).to(device)
    train_loader = make_loader(train_x, train_y, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, args.batch_size, False)
    test_loader = make_loader(test_x, test_y, args.batch_size, False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            target = yb[:, None].expand(-1, model.patches)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            ce = F.cross_entropy(logits.reshape(-1, classes), target.reshape(-1), weight=weights)
            loss = ce + args.tmse_weight * tmse_loss(logits, args.tmse_threshold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(yb)
            total_count += len(yb)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, device)
            if val[args.selection_metric] > best_val:
                best_val = val[args.selection_metric]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"{name} epoch={epoch} loss={total_loss / max(total_count, 1):.4f} "
                f"val_acc={val['accuracy']:.4f} val_fw={val['f1_weighted']:.4f}",
                flush=True,
            )

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    primary = ref["primary_metric"]
    gap = test[primary] - ref["paper_metrics"][primary]
    return {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": test[primary],
        "primary_metric": primary,
        "metrics": test,
        "source": "paper_style_reimplementation",
        "protocol_status": "paper_protocol_reimplementation",
        "official_code_used": False,
        "paper_reference_score": ref["paper_metrics"][primary],
        "paper_reference_metric": primary,
        "paper_reference_source": ref["source"],
        "paper_reference_gap_signed": float(gap),
        "paper_reference_gap_abs": float(abs(gap)),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range" if abs(gap) <= args.gap_threshold else "outside_expected_range",
        "calibration_status": "passed" if abs(gap) <= args.gap_threshold else "failed",
        "paper_metrics": ref["paper_metrics"],
        "dataset_preparation": meta,
        "runtime_seconds": round(time.time() - start, 3),
        "model_note": "P2LHAP paper-style reimplementation: non-overlap patching, shared channel-independent Transformer encoder, Transformer decoder, patch-label CE, and T-MSE smoothing.",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patch_size": args.patch_size,
            "d_model": args.d_model,
            "heads": args.heads,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "tmse_weight": args.tmse_weight,
            "tmse_threshold": args.tmse_threshold,
            "selection_metric": args.selection_metric,
            "seed": args.seed,
        },
    }


def make_entry(results: list[dict]) -> dict:
    datasets = []
    for result in results:
        ref = next(v for v in PAPER_REFERENCES.values() if v["benchmark_id"] == result["benchmark_id"])
        datasets.append({
            "benchmark_id": result["benchmark_id"],
            "dataset": ref["dataset"],
            "reported_effect": (
                f"P2LHAP paper-style rerun {result['primary_metric']} {result['primary_score']:.4f} "
                f"vs paper {result['paper_reference_score']:.4f}; gap {result['paper_reference_gap_abs']:.4f}."
            ),
            "protocol_note": ref["protocol_note"],
        })
    return {
        "id": "p2lhap",
        "method_name": "P2LHAP",
        "year": 2024,
        "method_family": "patch-to-label seq2seq transformer HAR",
        "submission_type": "paper_reimplementation",
        "paper_url": "https://arxiv.org/abs/2403.08214",
        "code_url": "",
        "coverage_status": "paper_protocol_calibration",
        "summary": "Patch-to-Label Seq2Seq Transformer for wearable HAR activity classification/segmentation/forecasting.",
        "implementation": "Local PyTorch paper-style reimplementation because no public author code was found.",
        "har_scope": "Wearable / IMU sensor-based human activity recognition.",
        "datasets": datasets,
        "results": results,
        "sources": ["https://arxiv.org/abs/2403.08214"],
    }


def select_device(args) -> torch.device:
    if args.device == "cuda":
        return torch.device(f"cuda:{args.gpu}")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_REFERENCES), default=["wisdm", "unimib_shar", "pamap2"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tmse-weight", type=float, default=0.15)
    parser.add_argument("--tmse-threshold", type=float, default=4.0)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--window-stride", type=int, default=100)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["accuracy", "f1_weighted", "f1_macro"], default="f1_weighted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--run-tag", default="p2lhap_paper_reimpl_20260727")
    args = parser.parse_args(argv)

    seed_all(args.seed)
    device = select_device(args)
    print(f"Using device: {device}", flush=True)
    results = [train_one_dataset(name, args, device) for name in args.datasets]
    out = RUNS_DIR / args.run_tag / "results.json"
    write_json(out, [make_entry(results)])
    print(f"Wrote {out}", flush=True)
    for r in results:
        print(
            f"{r['benchmark_id']} {r['primary_metric']}={r['primary_score']:.6f} "
            f"paper={r['paper_reference_score']:.6f} gap={r['paper_reference_gap_abs']:.6f} {r['calibration_status']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
