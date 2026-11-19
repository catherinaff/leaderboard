"""Prepare MHEALTH files for the official SensorLLM Stage-2 code.

This mirrors the public `mhealth_stage2.ipynb` notebook from the official
SensorLLM repository: 15 channels, subjects 1/3/6 as test, 100-sample windows,
50-sample stride, and an extra tail window when a segment is not exactly
covered by the stride.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "experiments" / "datasets"

ACTIVITY_MAP = {
    1: "Standing still (1 min)",
    2: "Sitting and relaxing (1 min)",
    3: "Lying down (1 min)",
    4: "Walking (1 min)",
    5: "Climbing stairs (1 min)",
    6: "Waist bends forward (20x)",
    7: "Frontal elevation of arms (20x)",
    8: "Knees bending (crouching) (20x)",
    9: "Cycling (1 min)",
    10: "Jogging (1 min)",
    11: "Running (1 min)",
    12: "Jump front & back (20x)",
}

COLS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 23]
NAMES = [
    "acc_ch_x",
    "acc_ch_y",
    "acc_ch_z",
    "acc_la_x",
    "acc_la_y",
    "acc_la_z",
    "gyr_la_x",
    "gyr_la_y",
    "gyr_la_z",
    "acc_rw_x",
    "acc_rw_y",
    "acc_rw_z",
    "gyr_rw_x",
    "gyr_rw_y",
    "gyr_rw_z",
    "activity",
]
TEST_SUBJECTS = {"subject1", "subject3", "subject6"}


def check_label_continuity(df: pd.DataFrame) -> dict[tuple[str, int], list[tuple[int, int]]]:
    continuity_segments: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for subject in df["subject"].unique():
        subject_data = df[df["subject"] == subject]
        assert subject_data.index[0] == 0
        for label in subject_data["activity"].unique():
            label_data = subject_data[subject_data["activity"] == label]
            indices = label_data.index
            segments = []
            start_idx = int(indices[0])
            for i in range(len(indices) - 1):
                if indices[i] + 1 != indices[i + 1]:
                    end_idx = int(indices[i])
                    segments.append((start_idx, end_idx))
                    start_idx = int(indices[i + 1])
            segments.append((start_idx, int(indices[-1])))
            if segments:
                continuity_segments[(subject, int(label))] = segments
    return continuity_segments


def split_sequences(sequences: list[list[float]], window_size: int, stride: int) -> tuple[list[np.ndarray], list[list[int]]]:
    assert len(sequences[0]) == 15
    has_null = any(any(pd.isnull(item) or item == "" for item in row) for row in sequences)
    if has_null:
        raise ValueError("Has null values")
    segments = []
    labels = []
    num_complete_segments = (len(sequences) - window_size) // stride + 1
    for i in range(num_complete_segments):
        start = i * stride
        end = start + window_size
        segment = sequences[start:end]
        assert len(segment) == window_size
        segments.append(np.array(segment))
        labels.append([start, end - 1])
    if labels and labels[-1][1] < len(sequences) - 1:
        start = len(sequences) - window_size
        end = len(sequences)
        segment = sequences[start:end]
        assert len(segment) == window_size
        segments.append(np.array(segment))
        labels.append([start, end - 1])
    return segments, labels


def summarize_channel(values: np.ndarray) -> str:
    trend = "steady"
    delta = float(values[-1] - values[0])
    if delta > 1e-6:
        trend = "increasing"
    elif delta < -1e-6:
        trend = "decreasing"
    return (
        f"mean {values.mean():.3f}, standard deviation {values.std():.3f}, "
        f"minimum {values.min():.3f}, maximum {values.max():.3f}, overall {trend} trend"
    )


def qa_gen(label: dict, data: np.ndarray, rng: random.Random) -> dict:
    channel_names = NAMES[:-1]
    picked = rng.sample(range(len(channel_names)), 2)
    trend_text = "; ".join(f"{channel_names[i]}: {summarize_channel(data[:, i])}" for i in picked)
    info_text = "; ".join(f"{name}: {summarize_channel(data[:, i])}" for i, name in enumerate(channel_names))
    question = "Please identify the human activity from the provided motion sensor readings."
    return {
        "Q": question,
        "A": label["activity_name"],
        "smry": f"The window contains MHEALTH readings from {label['subject']} for activity {label['activity_name']}.",
        "trend_text": trend_text,
        "corr_text": "",
        "info_text": info_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhealth-dir", default=str(DATASETS_DIR / "mhealth" / "MHEALTHDATASET"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "datasets" / "sensorllm_official" / "mhealth" / "whole_data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.mhealth_dir)
    output_dir = Path(args.output_dir)
    for split in ["train", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    split_segments = {"train": [], "test": []}
    split_labels = {"train": [], "test": []}
    file_rows = []

    for i in range(1, 11):
        df = pd.read_csv(data_dir / f"mHealth_subject{i}.log", header=None, sep=r"\s+")
        df = df.loc[:, COLS].rename(columns={c: n for c, n in zip(COLS, NAMES)})
        df["subject"] = f"subject{i}"
        continuity_segments = check_label_continuity(df)
        for key, segments_for_label in continuity_segments.items():
            subject_name, activity = key
            if activity == 0:
                continue
            for segment in segments_for_label:
                rows = df.loc[segment[0] : segment[1]]
                assert len(rows["subject"].unique()) == 1
                assert rows["subject"].unique()[0] == subject_name
                assert len(rows["activity"].unique()) == 1
                values = rows.iloc[:, ~rows.columns.isin(["subject", "activity"])].values.tolist()
                windows, spans = split_sequences(values, 100, 50)
                split = "test" if subject_name in TEST_SUBJECTS else "train"
                split_segments[split].extend(windows)
                for span in spans:
                    split_labels[split].append({
                        "subject": subject_name,
                        "activity_name": ACTIVITY_MAP[activity],
                        "activity": activity - 1,
                        "segments": span,
                    })
            file_rows.append({"subject": subject_name, "activity": int(activity), "segments": len(segments_for_label)})

    for split in ["train", "test"]:
        data_name = f"mhealth_{split}_data_stage2.pkl"
        labels_name = f"mhealth_{split}_labels_stage2.pkl"
        qa_name = f"mhealth_{split}_qa_stage2_cls.json"
        with open(output_dir / split / data_name, "wb") as f:
            pickle.dump(split_segments[split], f)
        with open(output_dir / split / labels_name, "wb") as f:
            pickle.dump(split_labels[split], f)
        qa_dict = {"author": "", "version": "", "date": str(datetime.now().date()), "dataset": []}
        for idx, (label, data) in enumerate(zip(split_labels[split], split_segments[split])):
            assert data.shape == (100, 15)
            qa_dict["dataset"].append({"index": idx, "qa_pair": qa_gen(label, data, rng)})
        with open(output_dir / split / qa_name, "w", encoding="utf-8") as f:
            json.dump(qa_dict, f, indent=2)

    manifest = {
        "source": "SensorLLM official mhealth_stage2.ipynb logic, scripted for reproducibility",
        "mhealth_dir": str(data_dir),
        "output_dir": str(output_dir),
        "train_windows": len(split_segments["train"]),
        "test_windows": len(split_segments["test"]),
        "channels": 15,
        "window_size": 100,
        "stride": 50,
        "test_subjects": sorted(TEST_SUBJECTS),
        "file_rows": file_rows,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
