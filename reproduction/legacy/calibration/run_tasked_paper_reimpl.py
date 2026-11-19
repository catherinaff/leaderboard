"""Paper-style TASKED reproduction.

No author code is available in this workspace. This runner implements the
TASKED paper's core training ingredients: a Transformer-like feature extractor,
weighted CE + Dice activity loss, subject discriminator, MMD regularization,
and self-knowledge distillation from a step-1 teacher. Dataset preparation
follows the paper protocols where the raw data are available.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
import types
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "experiments" / "datasets"
RUNS_DIR = ROOT / "experiments" / "runs"
DEEPCONVLSTM_OFFICIAL = ROOT / "experiments" / "repos" / "DeepConvLSTM_official"
GAP_THRESHOLD = 0.05

PAMAP2_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PAMAP2_LABEL = {activity: i for i, activity in enumerate(PAMAP2_ACTIVITIES)}

PAPER_REFERENCES = {
    "pamap2": {
        "benchmark_id": "pamap2",
        "dataset": "PAMAP2",
        "classes": 12,
        "paper_metrics": {"accuracy": 0.8304, "f1_weighted": 0.8293, "f1_macro": 0.7521},
        "primary_metric": "accuracy",
        "source": "TASKED paper Table 3, PAMAP2 row: Acc/Fw/Fm 83.04/82.93/75.21%.",
        "protocol_note": "PAMAP2 paper protocol: 8 protocol subjects, 12 activities, 36 channels, 100Hz, 200-sample windows, 50-sample step, LOSO with two validation subjects per test subject.",
    },
    "opportunity_b": {
        "benchmark_id": "opportunity_b",
        "dataset": "OPPORTUNITY Locomotion",
        "classes": 5,
        "paper_metrics": {"accuracy": 0.7583, "f1_weighted": 0.7583, "f1_macro": 0.7709},
        "primary_metric": "f1_weighted",
        "source": "TASKED paper Table 3, Opportunity locomotion row: Acc/Fw/Fm 75.83/75.83/77.09%.",
        "protocol_note": "OPPORTUNITY locomotion paper protocol: 4 subjects, 5 classes, 113 body-worn channels, 30Hz, 64-sample windows, 16-sample step, LOSO with two validation subjects per test subject.",
    },
    "mhealth": {
        "benchmark_id": "mhealth",
        "dataset": "MHEALTH",
        "classes": 12,
        "paper_metrics": {"accuracy": 0.95, "f1_weighted": 0.9466, "f1_macro": 0.9442},
        "primary_metric": "accuracy",
        "source": "TASKED paper Table 3, MHEALTH row: Acc/Fw/Fm 95.00/94.66/94.42%.",
        "protocol_note": "MHEALTH paper protocol: 10 subjects, 12 activities, 23 channels, 50Hz, 200-sample windows, 50-sample step, LOSO with two validation subjects per test subject.",
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


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    rows = np.arange(out.shape[0])
    for col in range(out.shape[1]):
        x = out[:, col]
        good = np.isfinite(x)
        if good.all():
            continue
        if not good.any():
            x[:] = 0.0
        else:
            x[~good] = np.interp(rows[~good], rows[good], x[good])
    return out


def sliding_windows(values: np.ndarray, labels: np.ndarray, subject: int, window: int, stride: int) -> tuple[list[np.ndarray], list[int], list[int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ss: list[int] = []
    if len(values) < window:
        return xs, ys, ss
    for start in range(0, len(values) - window + 1, stride):
        end = start + window
        ys_window = labels[start:end]
        counts = np.bincount(ys_window.astype(np.int64))
        label = int(counts.argmax())
        xs.append(values[start:end].astype(np.float32))
        ys.append(label)
        ss.append(subject)
    return xs, ys, ss


def pamap2_cols_36() -> list[int]:
    cols: list[int] = []
    for base in [3, 20, 37]:
        cols.extend(range(base + 1, base + 13))
    return cols


def load_pamap2() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "pamap2" / "PAMAP2_Dataset" / "Protocol"
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    cols = pamap2_cols_36()
    for path in sorted(data_dir.glob("subject*.dat")):
        subject_raw = int(path.stem.replace("subject", ""))
        if subject_raw == 109:
            continue
        subject = subject_raw - 100
        raw = np.loadtxt(path, dtype=np.float32)
        activity = raw[:, 1].astype(int)
        values = interpolate_nan(raw[:, cols])
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True) + 1e-6
        values = ((values - mean) / std).astype(np.float32)
        start = 0
        while start < len(values):
            act = int(activity[start])
            end = start + 1
            while end < len(values) and int(activity[end]) == act:
                end += 1
            if act in PAMAP2_LABEL:
                mapped = np.full(end - start, PAMAP2_LABEL[act], dtype=np.int64)
                x_part, y_part, s_part = sliding_windows(values[start:end], mapped, subject, 200, 50)
                xs.extend(x_part)
                ys.extend(y_part)
                subjects.extend(s_part)
            start = end
    if not xs:
        raise SystemExit("No PAMAP2 windows generated.")
    meta = {
        "raw_dir": str(data_dir),
        "subjects": sorted(set(subjects)),
        "window_size_samples": 200,
        "stride_samples": 50,
        "channels": "36 channels: per IMU acc16_xyz, acc6_xyz, gyro_xyz, magnetometer_xyz; heart-rate, temperature, and orientation removed.",
        "normalization": "per-subject channel z-score before windowing, matching TASKED paper text.",
    }
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(subjects, dtype=np.int64), meta


def load_py2_module(path: Path, module_name: str):
    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    source = path.read_text(encoding="utf-8")
    if not source.endswith("\n"):
        source += "\n"
    converted = str(tool.refactor_string(source, str(path)))
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.path.insert(0, str(path.parent))
    exec(compile(converted, str(path), "exec"), module.__dict__)
    return module


def load_opportunity_locomotion() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    preprocess_path = DEEPCONVLSTM_OFFICIAL / "preprocess_data.py"
    if not preprocess_path.exists():
        raise SystemExit(f"DeepConvLSTM official OPPORTUNITY preprocessing missing: {preprocess_path}")
    preprocess = load_py2_module(preprocess_path, "tasked_deepconvlstm_preprocess")
    data_root = DATASETS_DIR / "opportunity"
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    file_rows = []
    for relative_name in preprocess.OPPORTUNITY_DATA_FILES:
        path = data_root / relative_name
        if not path.exists():
            path = DATASETS_DIR / "opportunity" / "OpportunityUCIDataset" / relative_name
        raw = np.loadtxt(path)
        x_part, y_part = preprocess.process_dataset_file(raw, "locomotion")
        x_part = x_part.astype(np.float32)
        y_part = y_part.astype(np.int64)
        match = re.search(r"S(\d+)-", relative_name)
        subject = int(match.group(1)) if match else 0
        x_win, y_win, s_win = sliding_windows(x_part, y_part, subject, 64, 16)
        xs.extend(x_win)
        ys.extend(y_win)
        subjects.extend(s_win)
        file_rows.append({"file": relative_name, "samples": int(len(y_part)), "windows": len(y_win)})
    if not xs:
        raise SystemExit("No OPPORTUNITY locomotion windows generated.")
    y = np.asarray(ys, dtype=np.int64)
    # DeepConvLSTM preprocessing labels are 1-based for OPPORTUNITY.
    if y.min() == 1:
        y = y - 1
    meta = {
        "raw_dir": str(data_root),
        "subjects": sorted(set(subjects)),
        "window_size_samples": 64,
        "stride_samples": 16,
        "channels": "113 body-worn OPPORTUNITY channels from the DeepConvLSTM official preprocessing path.",
        "normalization": "official interpolation and fixed min/max normalization from DeepConvLSTM OPPORTUNITY preprocessing.",
        "file_rows": file_rows,
    }
    return np.asarray(xs, dtype=np.float32), y, np.asarray(subjects, dtype=np.int64), meta


def load_mhealth() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "mhealth" / "MHEALTHDATASET"
    if not data_dir.is_dir():
        alt = DATASETS_DIR / "mhealth"
        if (alt / "mHealth_subject1.log").exists():
            data_dir = alt
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    file_rows = []
    for path in sorted(data_dir.glob("mHealth_subject*.log")):
        match = re.search(r"subject(\d+)", path.name)
        subject = int(match.group(1)) if match else len(file_rows) + 1
        raw = np.loadtxt(path, dtype=np.float32)
        values = raw[:, :23]
        labels = raw[:, 23].astype(np.int64)
        keep = labels > 0
        values = values[keep]
        labels = labels[keep] - 1
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True) + 1e-6
        values = ((values - mean) / std).astype(np.float32)
        start = 0
        windows = 0
        while start < len(values):
            label = int(labels[start])
            end = start + 1
            while end < len(values) and int(labels[end]) == label:
                end += 1
            x_part, y_part, s_part = sliding_windows(values[start:end], labels[start:end], subject, 200, 50)
            xs.extend(x_part)
            ys.extend(y_part)
            subjects.extend(s_part)
            windows += len(y_part)
            start = end
        file_rows.append({"file": str(path), "windows": windows})
    if not xs:
        raise SystemExit(f"No MHEALTH windows generated under {data_dir}")
    meta = {
        "raw_dir": str(data_dir),
        "subjects": sorted(set(subjects)),
        "window_size_samples": 200,
        "stride_samples": 50,
        "channels": "23 sensor channels; label column removed.",
        "normalization": "per-subject channel z-score before windowing.",
        "file_rows": file_rows,
    }
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(subjects, dtype=np.int64), meta


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[int]]:
    if dataset == "pamap2":
        x, y, subjects, meta = load_pamap2()
        group_sizes = [12, 12, 12]
    elif dataset == "opportunity_b":
        x, y, subjects, meta = load_opportunity_locomotion()
        group_sizes = [x.shape[2]]
    elif dataset == "mhealth":
        x, y, subjects, meta = load_mhealth()
        group_sizes = [6, 9, 6, 2]
    else:
        raise KeyError(dataset)
    return x, y, subjects, meta, group_sizes


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class PositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-np.log(10000.0) / channels))
        pe = torch.zeros(max_len, channels)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.T.view(1, channels, max_len, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, : x.shape[2], :]


class TASKEDNet(nn.Module):
    def __init__(self, group_sizes: list[int], classes: int, domains: int, dropout: float) -> None:
        super().__init__()
        self.group_sizes = group_sizes
        self.sensor_convs = nn.ModuleList([nn.Conv1d(size, 32, kernel_size=3, padding=1) for size in group_sizes])
        self.pos = PositionalEncoding(32)
        self.blocks = nn.Sequential(
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, classes)
        self.discriminator = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, domains),
        )

    @staticmethod
    def _block(inp: int, out: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
        )

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = torch.split(x, self.group_sizes, dim=2)
        sensors = []
        for part, conv in zip(parts, self.sensor_convs):
            z = conv(part.transpose(1, 2))
            sensors.append(F.relu(z).unsqueeze(-1))
        z = torch.cat(sensors, dim=-1)
        z = self.pos(z)
        z = self.blocks(z)
        per_time = z.mean(dim=-1)
        pooled = per_time.mean(dim=-1)
        return pooled, per_time

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, per_time = self.features(x)
        logits = self.classifier(self.dropout(pooled))
        domain_logits = self.discriminator(grad_reverse(per_time, grl_lambda))
        return logits, domain_logits, pooled


def class_weights(y: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    return torch.from_numpy(counts.max() / counts)


def dice_loss(logits: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target = F.one_hot(labels, classes).float()
    inter = (probs * target).sum(dim=0)
    denom = probs.sum(dim=0) + target.sum(dim=0)
    return 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()


def activity_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=weights.to(logits.device))
    return 0.5 * ce + 0.5 * dice_loss(logits, labels, classes)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / tau, dim=1)
    t = F.softmax(teacher_logits / tau, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (tau * tau)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if len(x) < 2 or len(y) < 2:
        return x.new_tensor(0.0)
    z = torch.cat([x, y], dim=0)
    dist = torch.cdist(z, z).pow(2).detach()
    sigma = torch.median(dist[dist > 0]).clamp_min(1e-3)
    loss = x.new_tensor(0.0)
    for scale in [0.5, 1.0, 2.0, 4.0]:
        gamma = 1.0 / (2.0 * sigma * scale)
        kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
        kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
        kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
        loss = loss + kxx + kyy - 2.0 * kxy
    return loss / 4.0


def make_loader(x, y, subjects, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long(), torch.from_numpy(subjects).long()),
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


def evaluate(model: TASKEDNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb, _ in loader:
            logits, _, _ = model(xb.to(device), 0.0)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return metrics_for(np.concatenate(labels), np.concatenate(preds))


def domain_map(subjects: np.ndarray) -> dict[int, int]:
    return {subject: i for i, subject in enumerate(sorted({int(v) for v in subjects}))}


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    group_sizes: list[int],
    test_subject: int,
    val_subject: int,
    args,
    device: torch.device,
) -> dict:
    train_mask = (subjects != test_subject) & (subjects != val_subject)
    val_mask = subjects == val_subject
    test_mask = subjects == test_subject
    train_x, train_y, train_s = x[train_mask], y[train_mask], subjects[train_mask]
    val_x, val_y, val_s = x[val_mask], y[val_mask], subjects[val_mask]
    test_x, test_y, test_s = x[test_mask], y[test_mask], subjects[test_mask]
    dmap = domain_map(subjects)
    train_domains = np.asarray([dmap[int(v)] for v in train_s], dtype=np.int64)
    val_domains = np.asarray([dmap[int(v)] for v in val_s], dtype=np.int64)
    test_domains = np.asarray([dmap[int(v)] for v in test_s], dtype=np.int64)
    classes = int(y.max() + 1)
    domains = len(dmap)
    weights = class_weights(train_y, classes).to(device)
    train_loader = make_loader(train_x, train_y, train_domains, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, val_domains, args.batch_size, False)
    test_loader = make_loader(test_x, test_y, test_domains, args.batch_size, False)
    target_loader = make_loader(test_x, test_y, test_domains, args.batch_size, True)
    model = TASKEDNet(group_sizes, classes, domains, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    start = time.time()

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        for xb, yb, sb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, domain_logits, _ = model(xb, 0.0)
            loss = activity_loss(logits, yb, weights, classes) + F.cross_entropy(domain_logits, sb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
        if epoch % args.eval_every == 0 or epoch == args.pretrain_epochs:
            val = evaluate(model, val_loader, device)
            score = val[args.selection_metric]
            if score > best_val:
                best_val = score
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    disc_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    main_params = list(model.sensor_convs.parameters()) + list(model.blocks.parameters()) + list(model.classifier.parameters())
    main_opt = torch.optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    target_iter = iter(target_loader)

    for epoch in range(1, args.adv_epochs + 1):
        model.train()
        for xb, yb, sb in train_loader:
            try:
                xt, _, st = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, _, st = next(target_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            xt = xt.to(device)
            st = st.to(device)
            x_all = torch.cat([xb, xt], dim=0)
            s_all = torch.cat([sb, st], dim=0)
            disc_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, per_time = model.features(x_all)
            d_logits = model.discriminator(per_time.detach())
            d_loss = F.cross_entropy(d_logits, s_all)
            d_loss.backward()
            disc_opt.step()

            main_opt.zero_grad(set_to_none=True)
            logits, d_logits_grl, feat_train = model(xb, args.lambda_domain)
            with torch.no_grad():
                t_logits, _, _ = teacher(xb, 0.0)
            _, _, feat_target = model(xt, 0.0)
            loss_act = activity_loss(logits, yb, weights, classes)
            loss_kd = kd_loss(logits, t_logits, args.temperature)
            loss_cls = (1.0 - args.alpha) * loss_act + args.alpha * loss_kd
            loss_mmd = mmd_rbf(feat_train, feat_target)
            loss_domain = F.cross_entropy(d_logits_grl, sb)
            loss = args.lambda_cls * loss_cls + args.lambda_mmd * loss_mmd + args.lambda_domain * loss_domain
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, args.clip_grad)
            main_opt.step()
        if epoch % args.eval_every == 0 or epoch == args.adv_epochs:
            val = evaluate(model, val_loader, device)
            score = val[args.selection_metric]
            if score > best_val:
                best_val = score
                best_state = copy.deepcopy(model.state_dict())
            if args.verbose:
                print(f"fold test={test_subject} val={val_subject} epoch={epoch} val_{args.selection_metric}={score:.4f} best={best_val:.4f}", flush=True)

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    return {
        "test_subject": int(test_subject),
        "val_subject": int(val_subject),
        "train_windows": int(len(train_y)),
        "val_windows": int(len(val_y)),
        "test_windows": int(len(test_y)),
        "runtime_seconds": round(time.time() - start, 3),
        "best_val": float(best_val),
        "validation": val_metrics,
        "test": test_metrics,
    }


def summarize(rows: list[dict]) -> dict:
    keys = rows[0]["test"].keys()
    out = {}
    for key in keys:
        vals = np.asarray([row["test"][key] for row in rows], dtype=np.float64)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=0))
    return out


def validation_subjects(subjects: list[int], test_subject: int, count: int) -> list[int]:
    candidates = [s for s in subjects if s != test_subject]
    return candidates[:count]


def run_dataset(name: str, args, device: torch.device) -> dict:
    ref = PAPER_REFERENCES[name]
    x, y, subjects, meta, group_sizes = load_dataset(name)
    all_subjects = sorted({int(v) for v in subjects})
    rows = []
    fold_no = 0
    for test_subject in all_subjects:
        for val_subject in validation_subjects(all_subjects, test_subject, args.validation_subjects_per_test):
            fold_no += 1
            if args.max_folds and fold_no > args.max_folds:
                break
            seed_all(args.seed + fold_no)
            row = run_fold(x, y, subjects, group_sizes, test_subject, val_subject, args, device)
            rows.append(row)
            print(
                f"{name} fold={fold_no} test={test_subject} val={val_subject} "
                f"acc={row['test']['accuracy']:.4f} fw={row['test']['f1_weighted']:.4f} fm={row['test']['f1_macro']:.4f}",
                flush=True,
            )
        if args.max_folds and fold_no >= args.max_folds:
            break
    metrics = summarize(rows)
    primary = ref["primary_metric"]
    gap = metrics[primary] - ref["paper_metrics"][primary]
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": metrics[primary],
        "primary_metric": primary,
        "metrics": metrics,
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
        "fold_results": rows,
        "dataset_preparation": meta,
        "model_note": "Paper-style TASKED reimplementation: per-sensor Conv1D, positional encoding, temporal-spatial Conv2D attention surrogate, subject discriminator, MMD, gradient-reversal adversarial loss, and self-KD teacher.",
        "hyperparameters": {
            "pretrain_epochs": args.pretrain_epochs,
            "adv_epochs": args.adv_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_cls": args.lambda_cls,
            "lambda_mmd": args.lambda_mmd,
            "lambda_domain": args.lambda_domain,
            "alpha": args.alpha,
            "temperature": args.temperature,
        },
    }
    return result


def make_entry(results: list[dict]) -> dict:
    datasets = []
    for result in results:
        ref = next(v for v in PAPER_REFERENCES.values() if v["benchmark_id"] == result["benchmark_id"])
        datasets.append(
            {
                "benchmark_id": result["benchmark_id"],
                "dataset": ref["dataset"],
                "reported_effect": (
                    f"TASKED paper-style rerun {result['primary_metric']} {result['primary_score']:.4f} "
                    f"vs paper {result['paper_reference_score']:.4f}; gap {result['paper_reference_gap_abs']:.4f}."
                ),
                "protocol_note": ref["protocol_note"],
            }
        )
    passed = [r for r in results if r.get("calibration_status") == "passed"]
    return {
        "id": "tasked",
        "method_name": "TASKED",
        "year": 2022,
        "method_family": "transformer adversarial distillation HAR",
        "submission_type": "paper_reimplementation",
        "paper_url": "https://doi.org/10.1016/j.knosys.2022.110143",
        "coverage_status": "paper_protocol_calibration" if len(passed) >= 2 else "paper_protocol_partial_or_failed",
        "summary": "TASKED paper-style reimplementation against Table 3 reported scores.",
        "implementation": "Local PyTorch implementation of the paper's core TASKED components because no public author code was found in this workspace.",
        "har_scope": "Wearable HAR paper-protocol calibration before Standard promotion.",
        "datasets": datasets,
        "results": results,
        "sources": ["https://doi.org/10.1016/j.knosys.2022.110143"],
    }


def select_device(args) -> torch.device:
    if args.device == "cuda":
        return torch.device(f"cuda:{args.gpu}")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(PAPER_REFERENCES), default=["pamap2", "opportunity_b"])
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--adv-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda-cls", type=float, default=10.0)
    parser.add_argument("--lambda-mmd", type=float, default=5.0)
    parser.add_argument("--lambda-domain", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["accuracy", "f1_weighted", "f1_macro"], default="f1_weighted")
    parser.add_argument("--validation-subjects-per-test", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--run-tag", default="tasked_paper_reimpl_20260727")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    device = select_device(args)
    print(f"Using device: {device}", flush=True)
    results = [run_dataset(name, args, device) for name in args.datasets]
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
