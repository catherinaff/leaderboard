"""Run IF-ConvTransformer with the paper-table dataset protocols.

The public IF-ConvTransformer repository only ships ready-made HAPT and
OPPORTUNITY commands. This runner keeps the official model classes unchanged,
but rebuilds PAMAP2 and HHAR from raw files using the window/channel settings
reported in the paper comparison table instead of IoTBench canonical adapters.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
DATASETS_DIR = ROOT / "experiments" / "datasets"
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "IF-ConvTransformer-UbiComp2022"
OFFICIAL_REPO_URL = "https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022"
OFFICIAL_COMMIT = "577fa56"
GAP_THRESHOLD = 0.05

PAMAP2_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PAMAP2_LABELS = [
    "Lying",
    "Sitting",
    "Standing",
    "Walking",
    "Running",
    "Cycling",
    "Nordic Walking",
    "Ascending Stairs",
    "Descending Stairs",
    "Vacuum Cleaning",
    "Ironing",
    "Rope Jumping",
]
HHAR_LABELS = ["Biking", "Sitting", "Standing", "Walking", "Upstairs", "Downstairs"]
HHAR_LABEL_ALIASES = {
    "bike": 0,
    "biking": 0,
    "sit": 1,
    "sitting": 1,
    "stand": 2,
    "standing": 2,
    "walk": 3,
    "walking": 3,
    "stairsup": 4,
    "stairup": 4,
    "upstairs": 4,
    "stairs_up": 4,
    "stairsdown": 5,
    "stairdown": 5,
    "downstairs": 5,
    "stairs_down": 5,
}

PAPER_REFERENCES = {
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "paper_metrics": {"f1_macro": 0.9221, "f1_weighted": 0.9386},
        "paper_metric": "f1_weighted",
        "source": "IF-ConvTransformer IMWUT/UbiComp 2022 paper, Table IV, PAMAP2 row.",
        "protocol_note": "Paper table uses PAMAP2 with 9 participants, 12 activities, 3 IMUs, 100 Hz, window 48, overlap 24.",
    },
    "hhar": {
        "benchmark_id": "hhar",
        "dataset": "HHAR",
        "paper_metrics": {"f1_macro": 0.9501, "f1_weighted": 0.9513},
        "paper_metric": "f1_weighted",
        "source": "IF-ConvTransformer IMWUT/UbiComp 2022 paper, Table IV, HHAR row.",
        "protocol_note": "Paper table uses HHAR with 9 participants, 6 activities, variable-rate smartphone data unified by interpolation, window 200, overlap 100.",
    },
}


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        smooth_loss = -log_preds.sum(dim=-1) / classes
        nll = F.nll_loss(log_preds, target, reduction="none")
        return (self.eps * smooth_loss + (1.0 - self.eps) * nll).mean()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def install_import_shims() -> None:
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


def import_official_modules():
    src = OFFICIAL_REPO / "src"
    if not src.is_dir():
        raise SystemExit(f"Official IF-ConvTransformer repository missing: {src}")
    install_import_shims()
    sys.path.insert(0, str(src))
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from classifiers import If_ConvTransformer_W_torch, If_ConvTransformer_torch  # type: ignore
    finally:
        sys.argv = old_argv
    return If_ConvTransformer_torch, If_ConvTransformer_W_torch


def sliding_windows(values: np.ndarray, label: int, subject: int, window: int, stride: int) -> tuple[list[np.ndarray], list[int], list[int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ss: list[int] = []
    if len(values) < window:
        return xs, ys, ss
    for start in range(0, len(values) - window + 1, stride):
        xs.append(values[start : start + window].astype(np.float32))
        ys.append(label)
        ss.append(subject)
    return xs, ys, ss


def lowpass_gravity(acc: np.ndarray, fs: int, attitude_angles: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if len(acc) <= 18:
        grav = np.tile(np.mean(acc, axis=0, keepdims=True), (len(acc), 1))
    else:
        b, a = butter(3, 0.3 / (fs / 2), "low")
        grav = filtfilt(b, a, acc, axis=0)
    body = acc - grav
    if attitude_angles:
        angles = np.zeros_like(grav)
        angles[:, 0] = np.arctan2(grav[:, 1], grav[:, 2])
        angles[:, 1] = np.arctan2(grav[:, 0], grav[:, 2])
        angles[:, 2] = np.arctan2(grav[:, 0], grav[:, 1])
        return body.astype(np.float32), angles.astype(np.float32)
    return body.astype(np.float32), grav.astype(np.float32)


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


def pamap2_channel_indices() -> list[int]:
    starts = [3, 20, 37]
    cols: list[int] = []
    for start in starts:
        cols.extend(range(start + 1, start + 4))
        cols.extend(range(start + 7, start + 10))
    return cols


def load_pamap2(window: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "pamap2" / "PAMAP2_Dataset" / "Protocol"
    activity_to_label = {activity_id: i for i, activity_id in enumerate(PAMAP2_ACTIVITIES)}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    cols = pamap2_channel_indices()
    for path in sorted(data_dir.glob("subject*.dat")):
        subject = int(path.stem.replace("subject", ""))
        raw = np.loadtxt(path, dtype=np.float32)
        activities = raw[:, 1].astype(int)
        raw_channels = pd.DataFrame(raw[:, cols]).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
        per_imu = []
        for imu_idx in range(3):
            start_col = imu_idx * 6
            acc = raw_channels[:, start_col : start_col + 3]
            gyro = raw_channels[:, start_col + 3 : start_col + 6]
            body_acc, grav = lowpass_gravity(acc, fs=100, attitude_angles=True)
            per_imu.append(np.concatenate([grav, gyro, body_acc], axis=1))
        values = np.concatenate(per_imu, axis=1).astype(np.float32)
        start = 0
        while start < len(raw):
            activity = int(activities[start])
            end = start + 1
            while end < len(raw) and int(activities[end]) == activity:
                end += 1
            if activity in activity_to_label:
                x_part, y_part, s_part = sliding_windows(values[start:end], activity_to_label[activity], subject, window, stride)
                xs.extend(x_part)
                ys.extend(y_part)
                subjects.extend(s_part)
            start = end
    if not xs:
        raise SystemExit("No PAMAP2 paper-protocol windows were generated.")
    meta = {
        "paper_protocol": PAPER_REFERENCES["pamap2"]["protocol_note"],
        "raw_dir": str(data_dir),
        "subjects": sorted(set(subjects)),
        "activity_ids": PAMAP2_ACTIVITIES,
        "channel_order": "Per IMU: gravity attitude angles, gyroscope, body acceleration; IMUs: hand, chest, ankle.",
        "magnetometer_note": "PAMAP2 magnetometer columns are intentionally excluded because the official IF wearable preprocessing constructs 9 channels per IMU from accelerometer-derived gravity/body acceleration plus gyroscope.",
        "window_size_samples": window,
        "stride_samples": stride,
    }
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(subjects, dtype=np.int64), meta


def norm_hhar_label(value) -> int | None:
    key = str(value).strip().lower().replace(" ", "").replace("-", "_")
    if key in {"", "nan", "null"}:
        return None
    return HHAR_LABEL_ALIASES.get(key)


def load_hhar_csv(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["Creation_Time", "x", "y", "z", "User", "Device", "gt"]].copy()
    df["label"] = df["gt"].map(norm_hhar_label)
    df = df[df["label"].notna()]
    df["time"] = pd.to_numeric(df["Creation_Time"], errors="coerce")
    df = df[df["time"].notna()].sort_values("time")
    df["label"] = df["label"].astype(int)
    return df.rename(columns={"x": f"{prefix}_x", "y": f"{prefix}_y", "z": f"{prefix}_z"})[
        ["time", "User", "Device", "label", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    ]


def resample_hhar_group(merged: pd.DataFrame, hz: int) -> np.ndarray:
    merged = merged.sort_values("time")
    t = merged["time"].to_numpy(dtype=np.float64)
    if len(t) < 2:
        return np.empty((0, 6), dtype=np.float32)
    step_ns = 1_000_000_000.0 / hz
    target = np.arange(t[0], t[-1] + 1, step_ns, dtype=np.float64)
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    values = np.column_stack([np.interp(target, t, merged[c].to_numpy(dtype=np.float64)) for c in cols])
    return values.astype(np.float32)


def split_hhar_segments(merged: pd.DataFrame, max_gap_ns: int = 2_000_000_000) -> list[pd.DataFrame]:
    merged = merged.sort_values("time").reset_index(drop=True)
    if merged.empty:
        return []
    labels = merged["label"].to_numpy()
    times = merged["time"].to_numpy(dtype=np.float64)
    starts = [0]
    for i in range(1, len(merged)):
        if labels[i] != labels[i - 1] or times[i] - times[i - 1] > max_gap_ns:
            starts.append(i)
    starts.append(len(merged))
    return [merged.iloc[starts[i] : starts[i + 1]].copy() for i in range(len(starts) - 1)]


def load_hhar(window: int, stride: int, hz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "hhar" / "Activity recognition exp"
    acc = load_hhar_csv(data_dir / "Phones_accelerometer.csv", "acc")
    gyr = load_hhar_csv(data_dir / "Phones_gyroscope.csv", "gyro")
    users = {user: i + 1 for i, user in enumerate(sorted(acc["User"].dropna().unique()))}
    gyro_groups = {key: group.sort_values("time") for key, group in gyr.groupby(["User", "Device"], sort=False)}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    segment_count = 0
    for key, acc_group in acc.groupby(["User", "Device"], sort=False):
        gyro_group = gyro_groups.get(key)
        if gyro_group is None or gyro_group.empty:
            continue
        merged = pd.merge_asof(
            acc_group.sort_values("time"),
            gyro_group[["time", "gyro_x", "gyro_y", "gyro_z"]],
            on="time",
            direction="nearest",
            tolerance=50_000_000,
        ).dropna(subset=["gyro_x", "gyro_y", "gyro_z"])
        for segment in split_hhar_segments(merged):
            raw6 = resample_hhar_group(segment, hz)
            if len(raw6) < window:
                continue
            body_acc, grav = lowpass_gravity(raw6[:, :3], hz, attitude_angles=True)
            values = np.concatenate([grav, raw6[:, 3:6], body_acc], axis=1).astype(np.float32)
            label = int(segment["label"].iloc[0])
            x_part, y_part, s_part = sliding_windows(values, label, users[key[0]], window, stride)
            xs.extend(x_part)
            ys.extend(y_part)
            subjects.extend(s_part)
            segment_count += 1
    if not xs:
        raise SystemExit("No HHAR paper-protocol windows were generated.")
    meta = {
        "paper_protocol": PAPER_REFERENCES["hhar"]["protocol_note"],
        "raw_dir": str(data_dir),
        "subjects": sorted(set(subjects)),
        "window_size_samples": window,
        "stride_samples": stride,
        "resample_hz": hz,
        "segment_count": segment_count,
        "channel_order": "gravity attitude angles, gyroscope, body acceleration, matching the official smartphone model input layout.",
        "segmentation_note": "Accelerometer and gyroscope are aligned per User/Device stream, then split at label changes or time gaps before interpolation/windowing; this avoids stitching disjoint same-label recordings.",
    }
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(subjects, dtype=np.int64), meta


def to_model_input(x: np.ndarray) -> np.ndarray:
    return x[:, None, :, :].transpose(0, 1, 3, 2).astype(np.float32)


def split_train_val(y_train: np.ndarray, seed: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=y_train)
    return train_idx, val_idx


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device, non_blocking=True).float()
            logits = model(xb)[0]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return metrics_for(y, np.concatenate(preds))


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.maximum(lam, 1.0 - lam)
    lam_t = x.new_tensor(lam)
    shuffle = torch.randperm(batch_size, device=x.device)
    x2 = x[shuffle]
    y2 = y[shuffle]
    out_shape = [batch_size] + [1 for _ in range(x.dim() - 1)]
    mixed_x = x * lam_t.view(out_shape) + x2 * (1.0 - lam_t).view(out_shape)
    mixed_y = torch.cat([y[:, None].float(), y2[:, None].float(), lam_t[:, None].float()], dim=1)
    return mixed_x, mixed_y


def mixup_loss(logits: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return (criterion(logits, target[:, 0].long()) * target[:, 2] + criterion(logits, target[:, 1].long()) * (1 - target[:, 2])).mean()


def compute_center_loss(features: torch.Tensor, centers: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    features = features.view(features.size(0), -1)
    return F.mse_loss(features, centers[targets])


def get_center_delta(features: torch.Tensor, centers: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    delta = centers[targets] - features[indices]
    unique_targets, inverse = torch.unique(targets.cpu(), sorted=True, return_inverse=True)
    unique_targets = unique_targets.to(features.device)
    inverse = inverse.to(features.device)
    delta_centers = torch.zeros(unique_targets.size(0), delta.size(1), device=features.device).index_add_(0, inverse, delta)
    counts = torch.stack([(targets == label).sum() for label in unique_targets]).float().unsqueeze(1).to(features.device)
    delta_centers = delta_centers / (counts + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[unique_targets, :] = delta_centers
    return result


def make_model(dataset: str, input_channels: int, length: int, classes: int, device: torch.device) -> tuple[nn.Module, str]:
    smartphone_mod, wearable_mod = import_official_modules()
    if dataset == "pamap2":
        model = wearable_mod.If_ConvTransformer_W(1, input_channels, 64, 5, 3, 2, 128, 1, 0.2, "PAMAP2", length, classes)
        model.apply(wearable_mod.init_weights_orthogonal)
        return model.to(device), "If_ConvTransformer_W_torch"
    model = smartphone_mod.If_ConvTransformer(1, input_channels, 64, 5, 3, 2, 128, 1, 0.2, "HHAR", length, classes)
    return model.to(device), "If_ConvTransformer_torch"


def train_one_fold(dataset: str, x: np.ndarray, y: np.ndarray, subject: int, subjects: np.ndarray, args, device: torch.device) -> dict:
    seed_all(args.seed + int(subject))
    test_mask = subjects == subject
    train_mask = ~test_mask
    x_train_raw = x[train_mask]
    y_train_raw = y[train_mask]
    x_test_raw = x[test_mask]
    y_test = y[test_mask]
    train_idx, val_idx = split_train_val(y_train_raw, args.seed + int(subject), args.val_ratio)
    x_train_raw, x_val_raw, x_test_raw = standardize(x_train_raw[train_idx], x_train_raw[val_idx], x_test_raw)
    y_train = y_train_raw[train_idx]
    y_val = y_train_raw[val_idx]
    x_train = to_model_input(x_train_raw)
    x_val = to_model_input(x_val_raw)
    x_test = to_model_input(x_test_raw)
    model, model_name = make_model(dataset, x_train.shape[2], x_train.shape[-1], int(y.max() + 1), device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    if dataset == "pamap2":
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.wearable_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    else:
        criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.smartphone_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, min_lr=args.smartphone_lr / 10)

    best_state = None
    best_val = None
    best_key = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            if dataset == "pamap2":
                xb_mix, y_mix = mixup_data(xb, yb, args.mixup_alpha)
                logits, z = model(xb_mix)
                loss = mixup_loss(logits, y_mix, criterion)
                centers = model.centers
                loss = loss + args.center_beta * compute_center_loss(z, centers, yb)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.centers = centers - get_center_delta(z.detach(), centers, yb, args.center_lr)
            else:
                logits = model(xb)[0]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, x_val, y_val, args.batch_size, device)
        if val_metrics["accuracy"] > best_key:
            best_key = val_metrics["accuracy"]
            best_val = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dataset == "pamap2":
            scheduler.step()
        else:
            scheduler.step(1.0 - val_metrics["accuracy"])
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"{dataset} subject={subject} epoch={epoch} loss={np.mean(losses):.4f} val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, x_test, y_test, args.batch_size, device)
    return {
        "subject": int(subject),
        "metrics": test_metrics,
        "best_validation_metrics": best_val,
        "train_windows": int(len(x_train)),
        "val_windows": int(len(x_val)),
        "test_windows": int(len(x_test)),
        "runtime_seconds": round(time.time() - start_time, 3),
        "model": model_name,
    }


def aggregate_folds(folds: list[dict]) -> tuple[dict, dict]:
    keys = sorted(folds[0]["metrics"])
    mean = {key: float(np.mean([row["metrics"][key] for row in folds])) for key in keys}
    std = {key: float(np.std([row["metrics"][key] for row in folds])) for key in keys}
    return mean, std


def load_dataset(name: str, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if name == "pamap2":
        return load_pamap2(args.pamap2_window, args.pamap2_stride)
    return load_hhar(args.hhar_window, args.hhar_stride, args.hhar_hz)


def run_dataset(name: str, args, device: torch.device) -> dict:
    ref = PAPER_REFERENCES[name]
    x, y, subjects, meta = load_dataset(name, args)
    fold_results = []
    for subject in sorted(np.unique(subjects).tolist()):
        fold_results.append(train_one_fold(name, x, y, int(subject), subjects, args, device))
        last = fold_results[-1]
        print(
            "{} subject={} done f1_macro={:.4f} f1_weighted={:.4f} acc={:.4f}".format(
                name,
                subject,
                last["metrics"]["f1_macro"],
                last["metrics"]["f1_weighted"],
                last["metrics"]["accuracy"],
            ),
            flush=True,
        )
    metrics, metric_std = aggregate_folds(fold_results)
    gaps = {key: metrics[key] - value for key, value in ref["paper_metrics"].items()}
    max_abs_gap = max(abs(value) for value in gaps.values())
    primary = ref["paper_metric"]
    return {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "metric_std": metric_std,
        "source": "if_convtransformer_official_code_paper_protocol",
        "protocol_status": "official_code_paper_protocol",
        "paper_reference_metric": primary,
        "paper_reference_score": float(ref["paper_metrics"][primary]),
        "paper_reference_metrics": ref["paper_metrics"],
        "paper_reference_source": ref["source"],
        "paper_reference_note": ref["protocol_note"],
        "paper_reference_metric_gaps": gaps,
        "paper_reference_gap_signed": float(gaps[primary]),
        "paper_reference_gap_abs": float(abs(gaps[primary])),
        "paper_reference_max_gap_abs": float(max_abs_gap),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range" if max_abs_gap <= args.gap_threshold else "large_gap_check_protocol",
        "folds": fold_results,
        "fold_count": len(fold_results),
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": OFFICIAL_COMMIT,
        "data_loader_meta": meta,
        "input_channels": int(x.shape[2]),
        "classes": int(y.max() + 1),
        "samples": int(len(x)),
    }


def write_entry(results: list[dict], args) -> Path:
    datasets = []
    for row in results:
        ref = next(v for v in PAPER_REFERENCES.values() if v["benchmark_id"] == row["benchmark_id"])
        datasets.append(
            {
                "benchmark_id": row["benchmark_id"],
                "dataset": ref["dataset"],
                "activity_task": "Original IF-ConvTransformer paper dataset comparison.",
                "reported_effect": "{} paper-protocol rerun {} {:.4f} vs paper {:.4f}; max metric gap {:.4f}.".format(
                    ref["dataset"],
                    row["paper_reference_metric"],
                    row["primary_score"],
                    row["paper_reference_score"],
                    row["paper_reference_max_gap_abs"],
                ),
                "protocol_note": ref["protocol_note"],
            }
        )
    entry = {
        "id": "if_convtransformer",
        "method_name": "IF-ConvTransformer",
        "year": 2022,
        "method_family": "IMU fusion ConvTransformer",
        "submission_type": "platform",
        "paper_url": "https://dl.acm.org/doi/10.1145/3534584",
        "code_url": OFFICIAL_REPO_URL,
        "coverage_status": "official_code_paper_protocol",
        "summary": "IF-ConvTransformer paper-protocol rerun using raw PAMAP2/HHAR preprocessing aligned to the paper table window/channel settings.",
        "implementation": "Official IF-ConvTransformer PyTorch model classes imported unchanged; PAMAP2 uses wearable W branch, HHAR uses smartphone branch.",
        "har_scope": "IMU fusion ConvTransformer for smartphone and wearable HAR datasets.",
        "datasets": datasets,
        "results": results,
        "sources": [OFFICIAL_REPO_URL, "https://doi.org/10.1145/3534584"],
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
        raise SystemExit("CUDA requested but PyTorch cannot see CUDA.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    return device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_REFERENCES), default=sorted(PAPER_REFERENCES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smartphone-lr", type=float, default=0.0002)
    parser.add_argument("--wearable-lr", type=float, default=0.0005)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--center-beta", type=float, default=0.0003)
    parser.add_argument("--center-lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pamap2-window", type=int, default=48)
    parser.add_argument("--pamap2-stride", type=int, default=24)
    parser.add_argument("--hhar-window", type=int, default=200)
    parser.add_argument("--hhar-stride", type=int, default=100)
    parser.add_argument("--hhar-hz", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-tag", default="if_convtransformer_paper_protocol_pamap2_hhar_20260726")
    args = parser.parse_args(argv)

    seed_all(args.seed)
    device = select_device(args)
    print(f"Using device: {device}", flush=True)
    results = []
    for dataset in args.datasets:
        result = run_dataset(dataset, args, device)
        results.append(result)
        print(
            "{} done: {}={:.4f}, paper={:.4f}, max_gap={:.4f}, status={}".format(
                dataset,
                result["paper_reference_metric"],
                result["primary_score"],
                result["paper_reference_score"],
                result["paper_reference_max_gap_abs"],
                result["paper_reference_gap_status"],
            ),
            flush=True,
        )
    out_path = write_entry(results, args)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
