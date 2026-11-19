"""Preprocess WISDM into canonical sliding windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from har_utils import DATASETS_DIR, PROCESSED_DIR, load_benchmark, save_windows, windows_from_segments


RAW_PATH = DATASETS_DIR / "wisdm" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
OUT_PATH = PROCESSED_DIR / "wisdm_windows.npz"
LABELS = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
CHANNELS = ["acc_x", "acc_y", "acc_z"]


def iter_segments(raw_path: Path):
    current_key: tuple[int, int] | None = None
    rows: list[list[float]] = []

    def flush():
        nonlocal rows, current_key
        if current_key is not None and rows:
            subject, label = current_key
            yield subject, label, np.asarray(rows, dtype=np.float32)
        rows = []

    with raw_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            try:
                subject = int(parts[0])
                activity = parts[1]
                values = [float(parts[3]), float(parts[4]), float(parts[5])]
            except ValueError:
                continue
            if activity not in LABEL_TO_ID:
                continue

            key = (subject, LABEL_TO_ID[activity])
            if current_key is not None and key != current_key:
                yield from flush()
            current_key = key
            rows.append(values)

    yield from flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if OUT_PATH.is_file() and not args.force:
        print(f"Already processed: {OUT_PATH}")
        return 0
    if not RAW_PATH.is_file():
        raise SystemExit("WISDM raw data missing. Run experiments/download_wisdm.py first.")

    bench = load_benchmark("wisdm")
    protocol = bench["protocol"]
    x, y, subjects = windows_from_segments(
        iter_segments(RAW_PATH),
        window_size=int(protocol["window_size_samples"]),
        stride=int(protocol["stride_samples"]),
    )
    save_windows(
        OUT_PATH,
        x=x,
        y=y,
        subjects=subjects,
        benchmark_id="wisdm",
        label_names=LABELS,
        channel_names=CHANNELS,
        meta={
            "source_file": str(RAW_PATH.relative_to(DATASETS_DIR.parent.parent)),
            "protocol": protocol,
            "train_subjects": list(range(1, 31)),
            "test_subjects": list(range(31, 37)),
            "notes": "Windows are generated within contiguous subject/activity blocks; no window crosses an activity or subject boundary.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
