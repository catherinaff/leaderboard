"""Preprocess USC-HAD into canonical sliding windows."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from har_utils import DATASETS_DIR, PROCESSED_DIR, load_benchmark, save_windows, windows_from_segments


DATA_DIR = DATASETS_DIR / "usc_had"
OUT_PATH = PROCESSED_DIR / "usc_had_windows.npz"
LABELS = [
    "Walking Forward",
    "Walking Left",
    "Walking Right",
    "Walking Upstairs",
    "Walking Downstairs",
    "Running Forward",
    "Jumping Up",
    "Sitting",
    "Standing",
    "Sleeping",
    "Elevator Up",
    "Elevator Down",
]
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def find_sensor_array(mat: dict) -> np.ndarray:
    preferred = ["sensor_readings", "sensorData", "data"]
    for key in preferred:
        arr = mat.get(key)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and min(arr.shape) >= 6:
            return normalize_shape(arr)
    candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2 and min(value.shape) >= 6:
            candidates.append(value)
    if not candidates:
        raise ValueError("No 2D sensor array with at least 6 channels found in .mat file.")
    return normalize_shape(max(candidates, key=lambda a: max(a.shape)))


def normalize_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == 6 and arr.shape[1] > 6:
        arr = arr.T
    if arr.shape[1] < 6:
        raise ValueError(f"Expected at least 6 channels, got shape {arr.shape}")
    return arr[:, :6]


def iter_segments():
    mat_files = sorted(DATA_DIR.rglob("*.mat"))
    if not mat_files:
        raise SystemExit("USC-HAD .mat files missing. Run experiments/download_usc_had.py first.")
    for path in mat_files:
        subject_match = re.search(r"Subject\s*([0-9]+)|Subject([0-9]+)", str(path), re.IGNORECASE)
        activity_match = re.search(r"a([0-9]+)t[0-9]+", path.name, re.IGNORECASE)
        if not subject_match or not activity_match:
            continue
        subject = int(next(g for g in subject_match.groups() if g))
        activity = int(activity_match.group(1))
        if not (1 <= activity <= len(LABELS)):
            continue
        values = find_sensor_array(loadmat(path))
        yield subject, activity - 1, values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if OUT_PATH.is_file() and not args.force:
        print(f"Already processed: {OUT_PATH}")
        return 0

    bench = load_benchmark("usc_had")
    protocol = bench["protocol"]
    x, y, subjects = windows_from_segments(
        iter_segments(),
        window_size=int(protocol["window_size_samples"]),
        stride=int(protocol["stride_samples"]),
    )
    save_windows(
        OUT_PATH,
        x=x,
        y=y,
        subjects=subjects,
        benchmark_id="usc_had",
        label_names=LABELS,
        channel_names=CHANNELS,
        meta={
            "protocol": protocol,
            "notes": "Each trial .mat file is windowed independently; no window crosses a trial boundary.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
