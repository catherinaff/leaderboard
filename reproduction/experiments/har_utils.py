"""Shared utilities for local HAR benchmark reruns."""
from __future__ import annotations

import json
import math
import shutil
import ssl
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import certifi
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
DATASETS_DIR = ROOT / "experiments" / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"
RUNS_DIR = ROOT / "experiments" / "runs"


def load_benchmark(benchmark_id: str) -> dict:
    path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_file(url: str, dest: Path, *, chunk_size: int = 1024 * 1024, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        print(f"Already present: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url}")
    context = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "IoTBench-HAR/0.1"})
    with urllib.request.urlopen(req, context=context, timeout=60) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        last_report = time.time()
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total and time.time() - last_report > 5:
                print(f"  {done / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB")
                last_report = time.time()
    tmp.replace(dest)
    print(f"Wrote {dest}")
    return dest


def extract_archive(archive: Path, out_dir: Path, *, marker: Path | None = None, force: bool = False) -> None:
    if marker is not None and marker.exists() and not force:
        print(f"Already extracted: {out_dir}")
        return
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive} -> {out_dir}")
    lower = archive.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")


def fill_nan(x: np.ndarray) -> np.ndarray:
    """Fill NaN/inf values with per-channel finite means, then zeros."""
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).all():
        return x
    x = x.copy()
    finite = np.isfinite(x)
    means = np.nanmean(np.where(finite, x, np.nan), axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    bad = ~finite
    if bad.any():
        x[bad] = np.take(means, np.where(bad)[1])
    return x


def windows_from_segments(
    segments: Iterable[tuple[int, int, np.ndarray]],
    *,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[int] = []
    for subject, label, values in segments:
        values = fill_nan(values)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            xs.append(values[start : start + window_size])
            ys.append(label)
            subjects.append(subject)

    if not xs:
        raise RuntimeError("No windows were generated; check protocol and raw data layout.")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), np.asarray(subjects, dtype=np.int64)


def save_windows(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    benchmark_id: str,
    label_names: list[str],
    channel_names: list[str],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        subjects=subjects.astype(np.int64),
        label_names=np.asarray(label_names),
        channel_names=np.asarray(channel_names),
        benchmark_id=np.asarray(benchmark_id),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    print(f"Wrote {path}: x={x.shape}, classes={len(label_names)}, subjects={len(set(subjects.tolist()))}")


def load_windows(benchmark_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    path = PROCESSED_DIR / f"{benchmark_id}_windows.npz"
    if not path.is_file():
        raise SystemExit(f"Processed windows missing: {path}. Run experiments/preprocess_{benchmark_id}.py first.")
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["x"], data["y"], data["subjects"], meta


def window_features(x: np.ndarray) -> np.ndarray:
    """Compact deterministic statistical features for classical baselines."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    amin = x.min(axis=1)
    amax = x.max(axis=1)
    median = np.median(x, axis=1)
    q25 = np.percentile(x, 25, axis=1)
    q75 = np.percentile(x, 75, axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    feats = np.concatenate([mean, std, amin, amax, median, q75 - q25, rms], axis=1)
    return np.nan_to_num(feats, copy=False).astype(np.float32)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    keys = sorted({k for row in fold_metrics for k in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in fold_metrics], dtype=np.float64)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=0))
    return out


def majority_label(labels: np.ndarray) -> int:
    labels = labels.astype(np.int64)
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def subject_text(subjects: Iterable[int]) -> str:
    values = sorted({int(s) for s in subjects})
    if not values:
        return ""
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(v) for v in values)


def finite_or_none(value: float) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)
