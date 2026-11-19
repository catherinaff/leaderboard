"""Paper-style RobustHAR reproduction.

The IJCAI 2025 RobustHAR paper does not publish an author implementation in
the paper page. This runner implements the paper's core learning recipe:
location-inspired T x C x N sensor layout, sensor/segment/channel masked
pre-training with MAE reconstruction plus cosine consistency, and adaptive
multi-scale feature fusion for fine-tuning with 15% of the training labels.
"""
from __future__ import annotations

import argparse
import copy
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
RUNS_DIR = ROOT / "experiments" / "runs"
DATASETS_DIR = ROOT / "experiments" / "datasets"
GAP_THRESHOLD = 0.05

PAPER_REFERENCES = {
    "opportunity": {
        "benchmark_id": "opportunity_a",
        "dataset": "OPPORTUNITY",
        "paper_metrics": {"accuracy": 0.9372, "f1_weighted": 0.9374},
        "primary_metric": "f1_weighted",
        "source": "RobustHAR IJCAI 2025 Table 2, Opportunity row: Acc/F1 93.72/93.74%.",
        "protocol_note": "Paper split 8:1:1, pretrain 200 epochs batch 64 lr 0.001, finetune 100 epochs batch 32 lr 0.001, 15% labeled training data.",
        "default_sensors": 5,
    },
    "realworld": {
        "benchmark_id": "realworld_har",
        "dataset": "RealWorld HAR",
        "paper_metrics": {"accuracy": 0.9576, "f1_weighted": 0.9578},
        "primary_metric": "accuracy",
        "source": "RobustHAR IJCAI 2025 Table 2, RealWorld row: Acc/F1 95.76/95.78%.",
        "protocol_note": "Paper split 8:1:1, pretrain 200 epochs batch 64 lr 0.001, finetune 100 epochs batch 32 lr 0.001, 15% labeled training data.",
        "default_sensors": 7,
    },
    "czu_mhad": {
        "benchmark_id": "czu_mhad",
        "dataset": "CZU-MHAD",
        "paper_metrics": {"accuracy": 0.9463, "f1_weighted": 0.9469},
        "primary_metric": "accuracy",
        "source": "RobustHAR IJCAI 2025 Table 2, CZU-MHAD row: Acc/F1 94.63/94.69%.",
        "protocol_note": "Paper split 8:1:1, pretrain 200 epochs batch 64 lr 0.001, finetune 100 epochs batch 32 lr 0.001, 15% labeled training data.",
        "default_sensors": 10,
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


def ensure_ntf(x: np.ndarray, channel_count: int | None = None) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.shape}")
    if channel_count is not None:
        if x.shape[2] == channel_count:
            return x.astype(np.float32)
        if x.shape[1] == channel_count:
            return np.transpose(x, (0, 2, 1)).astype(np.float32)
        raise ValueError(f"Cannot infer time/features from shape {x.shape} and {channel_count=} channels")
    # Most prepared IoTBench arrays are stored as (samples, time, features).
    # Avoid guessing from relative sizes: OPPORTUNITY is (N, 30, 113).
    return x.astype(np.float32)


def normalize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


def split_8_1_1(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    train_x, tmp_x, train_y, tmp_y = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    val_x, test_x, val_y, test_y = train_test_split(
        tmp_x, tmp_y, test_size=0.50, random_state=seed, stratify=tmp_y
    )
    return train_x, train_y, val_x, val_y, test_x, test_y


def load_npz_split(path: Path, seed: int) -> tuple[np.ndarray, ...]:
    obj = np.load(path)
    keys = set(obj.files)
    channel_count = len(obj["channel_names"]) if "channel_names" in keys else None
    if {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"} <= keys:
        train_x = ensure_ntf(obj["X_train"], channel_count)
        train_y = obj["y_train"].astype(np.int64)
        val_x = ensure_ntf(obj["X_val"], channel_count)
        val_y = obj["y_val"].astype(np.int64)
        test_x = ensure_ntf(obj["X_test"], channel_count)
        test_y = obj["y_test"].astype(np.int64)
    elif {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"} <= keys:
        train_x = ensure_ntf(obj["x_train"], channel_count)
        train_y = obj["y_train"].astype(np.int64)
        val_x = ensure_ntf(obj["x_val"], channel_count)
        val_y = obj["y_val"].astype(np.int64)
        test_x = ensure_ntf(obj["x_test"], channel_count)
        test_y = obj["y_test"].astype(np.int64)
    elif {"X", "y"} <= keys or {"x", "y"} <= keys:
        x_key = "X" if "X" in keys else "x"
        x = ensure_ntf(obj[x_key], channel_count)
        y = obj["y"].astype(np.int64)
        train_x, train_y, val_x, val_y, test_x, test_y = split_8_1_1(x, y, seed)
    else:
        raise KeyError(f"{path} does not contain a supported split. Keys={sorted(keys)}")
    offset = min(int(train_y.min()), int(val_y.min()), int(test_y.min()))
    if offset != 0:
        train_y = train_y - offset
        val_y = val_y - offset
        test_y = test_y - offset
    return train_x, train_y, val_x, val_y, test_x, test_y


def synthetic_dataset(name: str, seed: int, sensors: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    classes = 8 if name != "opportunity" else 17
    n = 900 if name != "czu_mhad" else 500
    t = 64
    channels = 3
    y = np.arange(n, dtype=np.int64) % classes
    x = rng.normal(0, 0.35, size=(n, t, sensors * channels)).astype(np.float32)
    time = np.linspace(0, 2 * np.pi, t, dtype=np.float32)
    for i, cls in enumerate(y):
        freq = 1 + (cls % 5)
        amp = 0.8 + 0.12 * cls
        for s in range(sensors):
            base = s * channels
            x[i, :, base] += amp * np.sin(time * freq + s * 0.2)
            x[i, :, base + 1] += amp * np.cos(time * (freq + 0.5))
            x[i, :, base + 2] += (cls / max(classes - 1, 1)) + 0.05 * s
    return (*split_8_1_1(x, y, seed),)


def load_dataset(name: str, args) -> tuple[np.ndarray, ...]:
    if args.synthetic_smoke:
        sensors = args.sensors or PAPER_REFERENCES[name]["default_sensors"]
        data = synthetic_dataset(name, args.seed, sensors)
        meta = {"source": "synthetic_smoke", "note": "Synthetic data for runner validation only; not a reproduction score."}
    else:
        if not args.data_path:
            default = DATASETS_DIR / "robusthar" / f"{name}.npz"
            path = default
        else:
            path = Path(args.data_path)
        if not path.is_file():
            raise SystemExit(f"Prepared {name} npz not found: {path}")
        data = load_npz_split(path, args.seed)
        meta = {"source": str(path), "split": "existing split or stratified 8:1:1 from X/y npz"}
    train_x, train_y, val_x, val_y, test_x, test_y = data
    meta.update({
        "samples": {"train": int(len(train_y)), "val": int(len(val_y)), "test": int(len(test_y))},
        "shape": {"time_steps": int(train_x.shape[1]), "features": int(train_x.shape[2]), "classes": int(max(train_y.max(), val_y.max(), test_y.max()) + 1)},
    })
    train_x, val_x, test_x = normalize_by_train(train_x, val_x, test_x)
    return train_x, train_y, val_x, val_y, test_x, test_y, meta


def initial_position_matrix(sensors: int) -> torch.Tensor:
    return torch.eye(sensors, dtype=torch.float32)


class ScaleAutoencoder(nn.Module):
    def __init__(self, grid_features: int, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(grid_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        dec = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.decoder = nn.TransformerDecoder(dec, num_layers=1)
        self.query = nn.Parameter(torch.zeros(1, 512, d_model))
        self.out_proj = nn.Linear(d_model, grid_features)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        z = self.in_proj(x) + self.pos[:, :t]
        return self.encoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode_sequence(x)
        query = self.query[:, : x.shape[1]].expand(x.shape[0], -1, -1)
        rec = self.out_proj(self.decoder(query, memory))
        return rec, memory


class RobustHARNet(nn.Module):
    def __init__(self, features: int, classes: int, sensors: int, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if features % sensors != 0:
            grid_features = sensors * int(np.ceil(features / sensors))
        else:
            grid_features = features
        self.original_features = features
        self.features = grid_features
        self.sensors = sensors
        self.channels = grid_features // sensors
        self.register_buffer("position_init", initial_position_matrix(sensors))
        self.position_delta = nn.Parameter(torch.zeros(sensors, sensors))
        self.scales = nn.ModuleDict({
            "sensor": ScaleAutoencoder(grid_features, d_model, heads, dropout),
            "segment": ScaleAutoencoder(grid_features, d_model, heads, dropout),
            "channel": ScaleAutoencoder(grid_features, d_model, heads, dropout),
        })
        fusion_heads = max(1, min(heads, 4))
        self.sensor_cross = nn.MultiheadAttention(d_model, num_heads=fusion_heads, batch_first=True)
        self.channel_cross = nn.MultiheadAttention(d_model, num_heads=fusion_heads, batch_first=True)
        self.segment_cross = nn.MultiheadAttention(d_model, num_heads=fusion_heads, batch_first=True)
        self.fusion_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, classes))

    def position_matrix(self) -> torch.Tensor:
        logits = torch.log(self.position_init.clamp_min(1e-6)) + self.position_delta
        return torch.softmax(logits, dim=0)

    def to_lst3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.features:
            x = F.pad(x, (0, self.features - x.shape[-1]))
        elif x.shape[-1] > self.features:
            raise ValueError(f"Expected <= {self.features} features, got {x.shape[-1]}")
        data = x.view(x.shape[0], x.shape[1], self.sensors, self.channels)
        data = data.permute(0, 1, 3, 2).contiguous()
        return torch.einsum("btcn,nm->btcm", data, self.position_matrix())

    def flatten_lst3d(self, x3: torch.Tensor) -> torch.Tensor:
        return x3.reshape(x3.shape[0], x3.shape[1], self.channels * self.sensors)

    def mask(self, x3: torch.Tensor, kind: str, rate: float) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, time_steps, channels, _ = x3.shape
        mask = torch.ones_like(x3)
        if kind == "sensor":
            count = max(1, int(round(self.sensors * rate)))
            for b in range(bsz):
                idx = torch.randperm(self.sensors, device=x3.device)[:count]
                mask[b, :, :, idx] = 0
        elif kind == "channel":
            count = max(1, int(round(self.channels * rate)))
            idx = torch.randperm(channels, device=x3.device)[:count]
            mask[:, :, idx, :] = 0
        elif kind == "segment":
            span = max(1, int(round(time_steps * rate)))
            span = min(span, time_steps)
            for b in range(bsz):
                start = int(torch.randint(0, time_steps - span + 1, (1,), device=x3.device))
                mask[b, start : start + span, :] = 0
        else:
            raise KeyError(kind)
        return x3 * mask, mask

    def pretrain_step(self, x: torch.Tensor, mask_rate: float, lam: float) -> tuple[torch.Tensor, dict]:
        x3 = self.to_lst3d(x)
        flat_target = self.flatten_lst3d(x3)
        rec_loss = x.new_tensor(0.0)
        feats = []
        for kind, module in self.scales.items():
            masked3, mask3 = self.mask(x3, kind, mask_rate)
            masked = self.flatten_lst3d(masked3)
            rec, seq = module(masked)
            masked_region = self.flatten_lst3d(1.0 - mask3)
            denom = masked_region.sum().clamp_min(1.0)
            rec_loss = rec_loss + ((rec - flat_target).abs() * masked_region).sum() / denom
            feats.append(F.normalize(seq.mean(dim=1), dim=-1))
        con = x.new_tensor(0.0)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for left, right in pairs:
            con = con + (1.0 - F.cosine_similarity(feats[left], feats[right], dim=-1)).mean()
        con = con / len(pairs)
        return lam * rec_loss + (1.0 - lam) * con, {"rec_loss": float(rec_loss.detach().cpu()), "consistency_loss": float(con.detach().cpu())}

    def fused_features(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.flatten_lst3d(self.to_lst3d(x))
        fs = self.scales["sensor"].encode_sequence(flat)
        fse = self.scales["segment"].encode_sequence(flat)
        fc = self.scales["channel"].encode_sequence(flat)

        corr_sc = torch.bmm(F.normalize(fs, dim=-1), F.normalize(fc, dim=-1).transpose(1, 2))
        corr_cs = corr_sc.transpose(1, 2)
        rs = torch.softmax(corr_sc, dim=-1)
        rc = torch.softmax(corr_cs, dim=-1)
        as_seed = torch.bmm(rs, fs)
        ac_seed = torch.bmm(rc, fc)
        as_tokens, _ = self.sensor_cross(as_seed, fs, fs)
        ac_tokens, _ = self.channel_cross(ac_seed, fc, fc)
        fsc = as_tokens * ac_tokens

        corr_se = torch.bmm(F.normalize(fsc, dim=-1), F.normalize(fse, dim=-1).transpose(1, 2))
        segment_seed = torch.bmm(torch.softmax(corr_se, dim=-1), fse)
        segment_tokens, _ = self.segment_cross(segment_seed, fse, fse)
        fused = self.fusion_proj((fsc * segment_tokens) + fs + fc + fse)
        return self.norm(fused.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fused_features(x))


def make_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    tensor_x = torch.from_numpy(x).float()
    if y is None:
        ds = TensorDataset(tensor_x)
    else:
        ds = TensorDataset(tensor_x, torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, pin_memory=device.type == "cuda")


def select_labeled_subset(x: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return x, y
    idx = np.arange(len(y))
    keep, _ = train_test_split(idx, train_size=fraction, random_state=seed, stratify=y)
    return x[keep], y[keep]


def metrics_for(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def evaluate(model: RobustHARNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            preds.append(pred)
            labels.append(yb.numpy())
    return metrics_for(np.concatenate(labels), np.concatenate(preds))


def train_dataset(name: str, args, device: torch.device) -> dict:
    ref = PAPER_REFERENCES[name]
    train_x, train_y, val_x, val_y, test_x, test_y, meta = load_dataset(name, args)
    sensors = args.sensors or ref["default_sensors"]
    classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)
    model = RobustHARNet(train_x.shape[2], classes, sensors, args.d_model, args.heads, args.dropout).to(device)
    start = time.time()

    pre_loader = make_loader(train_x, None, args.pretrain_batch_size, True, device)
    pre_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        losses = []
        for (xb,) in pre_loader:
            xb = xb.to(device, non_blocking=True)
            pre_opt.zero_grad(set_to_none=True)
            loss, _ = model.pretrain_step(xb, args.mask_rate, args.balance_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            pre_opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_every == 0 or epoch == args.pretrain_epochs:
            print(f"{name} pretrain epoch={epoch} loss={np.mean(losses):.4f}", flush=True)

    labeled_x, labeled_y = select_labeled_subset(train_x, train_y, args.labeled_fraction, args.seed)
    train_loader = make_loader(labeled_x, labeled_y, args.finetune_batch_size, True, device)
    val_loader = make_loader(val_x, val_y, args.finetune_batch_size, False, device)
    test_loader = make_loader(test_x, test_y, args.finetune_batch_size, False, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    weights = torch.from_numpy(np.maximum(np.bincount(labeled_y, minlength=classes), 1).max() / np.maximum(np.bincount(labeled_y, minlength=classes), 1)).float().to(device)
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_every == 0 or epoch == args.finetune_epochs:
            val = evaluate(model, val_loader, device)
            if val[args.selection_metric] > best_val:
                best_val = val[args.selection_metric]
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{name} finetune epoch={epoch} loss={np.mean(losses):.4f} "
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
        "paper_reimplementation": True,
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
        "model_note": "Paper-style RobustHAR: T x C x N layout, sensor/segment/channel masked MAE pretraining, cosine feature consistency, and AMFF-like attention fusion.",
        "hyperparameters": {
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "pretrain_batch_size": args.pretrain_batch_size,
            "finetune_batch_size": args.finetune_batch_size,
            "lr": args.lr,
            "labeled_fraction": args.labeled_fraction,
            "balance_lambda": args.balance_lambda,
            "mask_rate": args.mask_rate,
            "sensors": sensors,
            "d_model": args.d_model,
            "heads": args.heads,
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
                f"RobustHAR paper-style rerun {result['primary_metric']} {result['primary_score']:.4f} "
                f"vs paper {result['paper_reference_score']:.4f}; gap {result['paper_reference_gap_abs']:.4f}."
            ),
            "protocol_note": ref["protocol_note"],
        })
    return {
        "id": "robusthar",
        "method_name": "RobustHAR",
        "year": 2025,
        "method_family": "masked self-supervised robust HAR",
        "submission_type": "paper_reimplementation",
        "paper_url": "https://www.ijcai.org/proceedings/2025/952",
        "code_url": "",
        "coverage_status": "paper_protocol_calibration" if results else "method_only_no_score",
        "summary": "Multi-scale spatial-temporal masked self-supervised pre-training for robust HAR under missing context.",
        "implementation": "Local PyTorch paper-style reimplementation because the IJCAI paper page does not publish author code.",
        "har_scope": "Wearable / IMU sensor-based human activity recognition.",
        "datasets": datasets,
        "results": results,
        "sources": ["https://www.ijcai.org/proceedings/2025/952"],
    }


def select_device(args) -> torch.device:
    if args.device == "cuda":
        return torch.device(f"cuda:{args.gpu}")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_REFERENCES), default=["opportunity", "realworld"])
    parser.add_argument("--data-path", default="")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--labeled-fraction", type=float, default=0.15)
    parser.add_argument("--balance-lambda", type=float, default=0.5)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--sensors", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--selection-metric", choices=["accuracy", "f1_weighted", "f1_macro"], default="f1_weighted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--run-tag", default="robusthar_paper_reimpl_20260727")
    args = parser.parse_args(argv)

    seed_all(args.seed)
    device = select_device(args)
    print(f"Using device: {device}", flush=True)
    results = [train_dataset(name, args, device) for name in args.datasets]
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
