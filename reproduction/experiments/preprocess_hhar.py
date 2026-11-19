"""Preprocess HHAR phone data into canonical sliding windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from har_utils import DATASETS_DIR, PROCESSED_DIR, load_benchmark, save_windows, windows_from_segments


DATA_DIR = DATASETS_DIR / "hhar" / "Activity recognition exp"
ACC_PATH = DATA_DIR / "Phones_accelerometer.csv"
GYR_PATH = DATA_DIR / "Phones_gyroscope.csv"
OUT_PATH = PROCESSED_DIR / "hhar_windows.npz"
LABELS = ["Biking", "Sitting", "Standing", "Walking", "Upstairs", "Downstairs"]
LABEL_ALIASES = {
    "bike": 0,
    "biking": 0,
    "sit": 1,
    "sitting": 1,
    "stand": 2,
    "standing": 2,
    "walk": 3,
    "walking": 3,
    "stairsup": 4,
    "stairsupstairs": 4,
    "stairup": 4,
    "stairupstairs": 4,
    "stairsup": 4,
    "stairs_up": 4,
    "upstairs": 4,
    "stairsdown": 5,
    "stairdown": 5,
    "stairdownstairs": 5,
    "stairsdown": 5,
    "stairs_down": 5,
    "downstairs": 5,
}
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
NS_PER_SAMPLE_50HZ = 20_000_000
MERGE_TOLERANCE_NS = 50_000_000


def norm_label(value) -> int | None:
    key = str(value).strip().lower().replace(" ", "").replace("-", "_")
    if key in {"", "nan", "null"}:
        return None
    return LABEL_ALIASES.get(key)


def load_phone_csv(path: Path, prefix: str) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"HHAR file missing: {path}. Run experiments/download_hhar.py first.")
    df = pd.read_csv(path)
    needed = ["Creation_Time", "x", "y", "z", "User", "Device", "gt"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} missing columns: {missing}")
    df = df[needed].copy()
    df["label"] = df["gt"].map(norm_label)
    df = df[df["label"].notna()]
    df["time"] = pd.to_numeric(df["Creation_Time"], errors="coerce")
    df = df[df["time"].notna()]
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"x": f"{prefix}_x", "y": f"{prefix}_y", "z": f"{prefix}_z"})
    return df[["time", "User", "Device", "label", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]]


def iter_segments():
    acc = load_phone_csv(ACC_PATH, "acc")
    gyr = load_phone_csv(GYR_PATH, "gyro")
    users = {user: i + 1 for i, user in enumerate(sorted(acc["User"].dropna().unique()))}
    gyr_groups = {
        key: group.sort_values("time")
        for key, group in gyr.groupby(["User", "Device", "label"], sort=False)
    }

    for key, acc_group in acc.groupby(["User", "Device", "label"], sort=False):
        gyro_group = gyr_groups.get(key)
        if gyro_group is None or gyro_group.empty:
            continue
        acc_group = acc_group.sort_values("time")
        merged = pd.merge_asof(
            acc_group,
            gyro_group[["time", "gyro_x", "gyro_y", "gyro_z"]],
            on="time",
            direction="nearest",
            tolerance=MERGE_TOLERANCE_NS,
        ).dropna(subset=["gyro_x", "gyro_y", "gyro_z"])
        if merged.empty:
            continue

        t0 = float(merged["time"].iloc[0])
        merged["bucket"] = ((merged["time"] - t0) // NS_PER_SAMPLE_50HZ).astype("int64")
        sampled = merged.groupby("bucket", sort=True).first()
        values = sampled[CHANNELS].to_numpy(dtype=np.float32)
        subject = users[key[0]]
        label = int(key[2])
        yield subject, label, values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if OUT_PATH.is_file() and not args.force:
        print(f"Already processed: {OUT_PATH}")
        return 0

    bench = load_benchmark("hhar")
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
        benchmark_id="hhar",
        label_names=LABELS,
        channel_names=CHANNELS,
        meta={
            "protocol": protocol,
            "notes": "Uses phone accelerometer and gyroscope files only. Streams are nearest-neighbor aligned by Creation_Time within 50 ms and downsampled to 50 Hz before windowing.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
