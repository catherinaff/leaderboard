"""Preprocess PAMAP2 into canonical sliding windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from har_utils import DATASETS_DIR, PROCESSED_DIR, load_benchmark, save_windows, windows_from_segments


DATA_DIR = DATASETS_DIR / "pamap2" / "PAMAP2_Dataset" / "Protocol"
OUT_PATH = PROCESSED_DIR / "pamap2_windows.npz"
CANONICAL_SUBJECTS = list(range(101, 109))
ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
LABELS = [
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
ACTIVITY_TO_LABEL = {activity_id: i for i, activity_id in enumerate(ACTIVITY_IDS)}


def channel_indices() -> list[int]:
    # PAMAP2 columns are 0-based here: timestamp, activity, heart-rate,
    # then hand/chest/ankle IMUs with temperature, acc16, acc6, gyro, mag, orientation.
    starts = [3, 20, 37]
    cols: list[int] = []
    for start in starts:
        cols.extend(range(start + 1, start + 4))    # 16g accelerometer
        cols.extend(range(start + 7, start + 10))   # gyroscope
        cols.extend(range(start + 10, start + 13))  # magnetometer
    return cols


CHANNELS = [
    f"{placement}_{axis}"
    for placement in ["hand", "chest", "ankle"]
    for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
]


def iter_segments():
    cols = channel_indices()
    files = sorted(DATA_DIR.glob("subject*.dat"))
    if not files:
        raise SystemExit("PAMAP2 protocol files missing. Run experiments/download_pamap2.py first.")

    for path in files:
        subject = int(path.stem.replace("subject", ""))
        if subject not in CANONICAL_SUBJECTS:
            continue
        data = np.loadtxt(path, dtype=np.float32)
        activities = data[:, 1].astype(np.int64)
        values = data[:, cols]

        start = 0
        while start < len(data):
            activity = int(activities[start])
            end = start + 1
            while end < len(data) and int(activities[end]) == activity:
                end += 1
            if activity in ACTIVITY_TO_LABEL:
                yield subject, ACTIVITY_TO_LABEL[activity], values[start:end]
            start = end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if OUT_PATH.is_file() and not args.force:
        print(f"Already processed: {OUT_PATH}")
        return 0

    bench = load_benchmark("pamap2")
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
        benchmark_id="pamap2",
        label_names=LABELS,
        channel_names=CHANNELS,
        meta={
            "protocol": protocol,
            "subjects": CANONICAL_SUBJECTS,
            "activity_ids": ACTIVITY_IDS,
            "notes": "Uses Protocol recordings only, excludes activity 0 and optional/low-count activities, and uses the 16g accelerometer, gyroscope, and magnetometer channels for each IMU.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
