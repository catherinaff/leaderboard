"""Preprocess OPPORTUNITY ADL runs into canonical sliding windows."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from har_utils import DATASETS_DIR, PROCESSED_DIR, load_benchmark, save_windows, windows_from_segments


DATA_DIR = DATASETS_DIR / "opportunity" / "OpportunityUCIDataset" / "dataset"
GESTURE_LABELS = {
    406516: 0,
    406517: 1,
    404516: 2,
    404517: 3,
    406520: 4,
    404520: 5,
    406505: 6,
    404505: 7,
    406519: 8,
    404519: 9,
    406511: 10,
    404511: 11,
    406508: 12,
    404508: 13,
    408512: 14,
    407521: 15,
    405506: 16,
}
GESTURE_NAMES = [
    "Open Door 1",
    "Open Door 2",
    "Close Door 1",
    "Close Door 2",
    "Open Fridge",
    "Close Fridge",
    "Open Dishwasher",
    "Close Dishwasher",
    "Open Drawer 1",
    "Close Drawer 1",
    "Open Drawer 2",
    "Close Drawer 2",
    "Open Drawer 3",
    "Close Drawer 3",
    "Clean Table",
    "Drink from Cup",
    "Toggle Switch",
]
LOCOMOTION_LABELS = {1: 0, 2: 1, 4: 2, 5: 3}
LOCOMOTION_NAMES = ["Stand", "Walk", "Sit", "Lie"]
CHANNELS = [f"opp_ch_{i:03d}" for i in range(113)]


def select_challenge_columns(data: np.ndarray) -> np.ndarray:
    """Select the 113 sensor channels and locomotion/gesture labels used in common OPPORTUNITY preprocessing."""
    delete = np.arange(46, 50)
    for start, stop in [(59, 63), (72, 76), (85, 89), (98, 102), (134, 243), (244, 249)]:
        delete = np.concatenate([delete, np.arange(start, stop)])
    return np.delete(data, delete, axis=1)


def iter_segments(track: str):
    files = sorted(DATA_DIR.glob("S*-ADL*.dat"))
    if not files:
        raise SystemExit("OPPORTUNITY ADL files missing. Run experiments/download_opportunity.py first.")
    label_map = GESTURE_LABELS if track == "a" else LOCOMOTION_LABELS

    for path in files:
        subject_match = re.search(r"S([0-9]+)-ADL", path.name)
        if not subject_match:
            continue
        subject = int(subject_match.group(1))
        raw = np.loadtxt(path, dtype=np.float32)
        selected = select_challenge_columns(raw)
        values = selected[:, 1:114]
        labels = selected[:, 115 if track == "a" else 114].astype(np.int64)

        start = 0
        while start < len(labels):
            original = int(labels[start])
            end = start + 1
            while end < len(labels) and int(labels[end]) == original:
                end += 1
            if original in label_map:
                yield subject, label_map[original], values[start:end]
            start = end


def preprocess(track: str, force: bool) -> None:
    benchmark_id = "opportunity_a" if track == "a" else "opportunity_b"
    out_path = PROCESSED_DIR / f"{benchmark_id}_windows.npz"
    if out_path.is_file() and not force:
        print(f"Already processed: {out_path}")
        return
    bench = load_benchmark(benchmark_id)
    protocol = bench["protocol"]
    x, y, subjects = windows_from_segments(
        iter_segments(track),
        window_size=int(protocol["window_size_samples"]),
        stride=int(protocol["stride_samples"]),
    )
    save_windows(
        out_path,
        x=x,
        y=y,
        subjects=subjects,
        benchmark_id=benchmark_id,
        label_names=GESTURE_NAMES if track == "a" else LOCOMOTION_NAMES,
        channel_names=CHANNELS,
        meta={
            "protocol": protocol,
            "notes": "Uses ADL runs only and the common 113-channel OPPORTUNITY challenge feature selection. Null/unannotated labels are excluded before windowing.",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    tracks = ["a", "b"] if args.track == "both" else [args.track]
    for track in tracks:
        preprocess(track, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
