"""Run UniMTS official fine-tuning code on leaderboard datasets.

This adapter prepares canonical leaderboard data in the official UniMTS
`finetune_custom.py` format, executes the upstream script, parses its reported
metrics, and writes a mergeable leaderboard result. The model implementation and
fine-tuning loop remain the official repository code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from har_utils import DATASETS_DIR, RUNS_DIR, load_benchmark, load_windows, summarize_fold_metrics, write_json


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "unimts"
VENDOR_STUBS = ROOT / "experiments" / "vendor_stubs"
OFFICIAL_REPO_URL = "https://github.com/xiyuanzh/unimts"
OFFICIAL_COMMIT = "656fcdab7eee17d5b4158acf98fa348f74ce29a6"
PYTHON = "/opt/conda/envs/iotbench-har/bin/python"
CUSTOM_DIR = DATASETS_DIR / "unimts_custom"
PAPER_REFERENCE = {
    "uci_har": {
        "score": 0.921,
        "metric": "accuracy",
        "source": "UniMTS paper Table 2 full-shot result",
        "protocol_note": "Paper reports full-shot UCIHAR with the official UniMTS dataset preparation.",
    },
    "pamap2": {
        "score": 0.981,
        "metric": "f1_weighted",
        "source": "UniMTS paper Table 2 full-shot result",
        "protocol_note": "Paper reports full-shot PAMAP with the official UniMTS dataset preparation.",
    },
    "usc_had": {
        "score": 0.631,
        "metric": "f1_macro",
        "source": "UniMTS paper Table 2 full-shot result",
        "protocol_note": "Paper reports full-shot USCHAD F1 with the official UniMTS dataset preparation; Table 2 accuracy is 58.3.",
    },
    "wisdm": {
        "score": 0.809,
        "metric": "f1_macro",
        "source": "UniMTS paper Table 2 full-shot result",
        "protocol_note": "Paper reports full-shot WISDM with an 18-class protocol; leaderboard WISDM is a 6-class variant.",
        "directly_comparable": False,
    },
}
GAP_THRESHOLD = 0.05
UCI_SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def load_uci_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = DATASETS_DIR / "uci_har" / "UCI HAR Dataset"
    sig_dir = data_dir / split / "Inertial Signals"
    if not sig_dir.is_dir():
        raise SystemExit("UCI-HAR inertial signal data missing. Run experiments/download_uci_har.py first.")
    arrays = [np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32) for name in UCI_SIGNALS]
    x = np.stack(arrays, axis=2)
    y = np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=np.int64) - 1

    # UniMTS official UCIHAR loader maps total_acc + body_gyro onto skeleton
    # joint 9 and converts acceleration from g to m/s^2 before resampling.
    x_custom = np.concatenate([x[:, :, 6:9] * 9.80665, x[:, :, 3:6]], axis=2).astype(np.float32)
    return x_custom, y.astype(np.int64)


def class_labels(benchmark_id: str) -> list[str]:
    bench = load_benchmark(benchmark_id)
    return [str(x).lower().replace("_", " ") for x in bench.get("classes", [])]


def prepare_uci_har(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_uci_split("train")
    x_test, y_test = load_uci_split("test")
    np.save(out_dir / "X_train.npy", x_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", x_test)
    np.save(out_dir / "y_test.npy", y_test)
    labels = class_labels("uci_har")
    write_json(
        out_dir / "UCIHAR.json",
        {"label_dictionary": {str(i): [label] for i, label in enumerate(labels)}},
    )
    split_dir = out_dir / "official_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]:
        shutil.copy2(out_dir / name, split_dir / name)
    return {
        "benchmark_id": "uci_har",
        "unimts_dataset_name": "UCIHAR",
        "custom_dir": str(out_dir),
        "config_path": str(out_dir / "UCIHAR.json"),
        "joint_list": [9],
        "original_sampling_rate": 50,
        "num_class": len(labels),
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "input_channels": "total_acc_xyz_mps2 + body_gyro_xyz",
        "splits": [
            {
                "name": "official_fixed",
                "train_windows": int(len(y_train)),
                "test_windows": int(len(y_test)),
                "dir": str(split_dir),
            }
        ],
    }


def load_processed(benchmark_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x, y, subjects, meta = load_windows(benchmark_id)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(subjects, dtype=np.int64), meta


def write_split(split_dir: Path, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "X_train.npy", x_train.astype(np.float32))
    np.save(split_dir / "y_train.npy", y_train.astype(np.int64))
    np.save(split_dir / "X_test.npy", x_test.astype(np.float32))
    np.save(split_dir / "y_test.npy", y_test.astype(np.int64))


def write_config(out_dir: Path, benchmark_id: str, name: str) -> Path:
    labels = class_labels(benchmark_id)
    path = out_dir / f"{name}.json"
    write_json(path, {"label_dictionary": {str(i): [label] for i, label in enumerate(labels)}})
    return path


def fixed_subject_indices(subjects: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    train_subjects = {int(x) for x in meta.get("train_subjects", [])}
    test_subjects = {int(x) for x in meta.get("test_subjects", [])}
    if not train_subjects or not test_subjects:
        raise SystemExit("Fixed split dataset needs train_subjects/test_subjects metadata.")
    return np.isin(subjects, sorted(train_subjects)), np.isin(subjects, sorted(test_subjects))


def make_splits(
    out_dir: Path,
    x_custom: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    meta: dict,
    protocol_kind: str,
) -> list[dict]:
    splits = []
    if protocol_kind == "fixed":
        train_idx, test_idx = fixed_subject_indices(subjects, meta)
        split_dir = out_dir / "fixed_subject_split"
        write_split(split_dir, x_custom[train_idx], y[train_idx], x_custom[test_idx], y[test_idx])
        return [
            {
                "name": "fixed_subject_split",
                "train_windows": int(train_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "dir": str(split_dir),
            }
        ]
    if protocol_kind != "loso":
        raise SystemExit(f"Unsupported UniMTS split kind: {protocol_kind}")
    for subject in sorted({int(s) for s in subjects}):
        test_idx = subjects == subject
        train_idx = ~test_idx
        split_dir = out_dir / f"loso_subject_{subject}"
        write_split(split_dir, x_custom[train_idx], y[train_idx], x_custom[test_idx], y[test_idx])
        splits.append(
            {
                "name": f"loso_subject_{subject}",
                "test_subject": int(subject),
                "train_windows": int(train_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "dir": str(split_dir),
            }
        )
    return splits


def prepare_wisdm(out_dir: Path) -> dict:
    x, y, subjects, meta = load_processed("wisdm")
    gyro_pad = np.zeros((*x.shape[:2], 3), dtype=np.float32)
    x_custom = np.concatenate([x[:, :, :3], gyro_pad], axis=2)
    config_path = write_config(out_dir, "wisdm", "WISDM")
    splits = make_splits(out_dir, x_custom, y, subjects, meta, "fixed")
    return {
        "benchmark_id": "wisdm",
        "unimts_dataset_name": "WISDM",
        "custom_dir": str(out_dir),
        "config_path": str(config_path),
        "joint_list": [21],
        "original_sampling_rate": 20,
        "num_class": len(class_labels("wisdm")),
        "input_channels": "smartphone_acc_xyz + zero gyro pad; UniMTS gyro=0 uses accelerometer channels",
        "splits": splits,
    }


def prepare_usc_had(out_dir: Path) -> dict:
    x, y, subjects, meta = load_processed("usc_had")
    x_custom = np.concatenate([x[:, :, :3] * 9.80665, x[:, :, 3:6] / 180.0 * np.pi], axis=2)
    config_path = write_config(out_dir, "usc_had", "USCHAD")
    splits = make_splits(out_dir, x_custom, y, subjects, meta, "loso")
    return {
        "benchmark_id": "usc_had",
        "unimts_dataset_name": "USCHAD",
        "custom_dir": str(out_dir),
        "config_path": str(config_path),
        "joint_list": [5],
        "original_sampling_rate": 100,
        "num_class": len(class_labels("usc_had")),
        "input_channels": "USC-HAD acc_xyz converted g->m/s^2 + gyro_xyz converted deg/s->rad/s",
        "splits": splits,
    }


def prepare_pamap2(out_dir: Path) -> dict:
    x, y, subjects, meta = load_processed("pamap2")
    hand = x[:, :, 0:6]
    chest = x[:, :, 9:15]
    ankle = x[:, :, 18:24]
    x_custom = np.concatenate([hand, ankle, chest], axis=2)
    config_path = write_config(out_dir, "pamap2", "PAMAP")
    splits = make_splits(out_dir, x_custom, y, subjects, meta, "loso")
    return {
        "benchmark_id": "pamap2",
        "unimts_dataset_name": "PAMAP",
        "custom_dir": str(out_dir),
        "config_path": str(config_path),
        "joint_list": [21, 11, 7],
        "original_sampling_rate": 100,
        "num_class": len(class_labels("pamap2")),
        "input_channels": "PAMAP2 hand, ankle, chest acc_xyz+gyro_xyz mapped to UniMTS joints [21, 11, 7]",
        "splits": splits,
        "primary_metric_note": "UniMTS official script reports macro F1 by default; weighted-F1 parsing must be enabled before merging PAMAP2 as a primary leaderboard row.",
    }


PREPARERS = {
    "uci_har": prepare_uci_har,
    "wisdm": prepare_wisdm,
    "usc_had": prepare_usc_had,
    "pamap2": prepare_pamap2,
}


def prepare_dataset(benchmark_id: str) -> dict:
    preparer = PREPARERS.get(benchmark_id)
    if preparer is None:
        raise SystemExit(f"UniMTS official adapter does not yet support {benchmark_id}.")
    return preparer(CUSTOM_DIR / benchmark_id)


def run_official(meta: dict, split: dict, args) -> dict:
    if not OFFICIAL_REPO.is_dir():
        raise SystemExit(f"Official UniMTS repository missing: {OFFICIAL_REPO}")

    cmd = [
        PYTHON,
        "finetune_custom.py",
        "--mode",
        args.mode,
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.epochs),
        "--checkpoint",
        "./checkpoint/UniMTS.pth",
        "--X_train_path",
        str(Path(split["dir"]) / "X_train.npy"),
        "--y_train_path",
        str(Path(split["dir"]) / "y_train.npy"),
        "--X_test_path",
        str(Path(split["dir"]) / "X_test.npy"),
        "--y_test_path",
        str(Path(split["dir"]) / "y_test.npy"),
        "--config_path",
        str(meta["config_path"]),
        "--joint_list",
        *[str(x) for x in meta["joint_list"]],
        "--original_sampling_rate",
        str(meta["original_sampling_rate"]),
        "--num_class",
        str(meta["num_class"]),
        "--run_tag",
        f"{args.run_tag}_{split['name']}",
        "--stage",
        "finetune",
    ]
    if args.gyro:
        cmd.extend(["--gyro", "1"])

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    env["PYTHONPATH"] = str(VENDOR_STUBS) + os.pathsep + env.get("PYTHONPATH", "")

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=OFFICIAL_REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip())
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"UniMTS official finetune_custom.py failed with exit code {code}")

    metric_line = next((line for line in reversed(lines) if line.startswith("acc: ")), "")
    match = re.search(
        r"acc:\s*([0-9.eE+-]+),\s*prec:\s*([0-9.eE+-]+),\s*rec:\s*([0-9.eE+-]+),\s*f1:\s*([0-9.eE+-]+)(?:,\s*f1_weighted:\s*([0-9.eE+-]+))?",
        metric_line,
    )
    if not match:
        raise SystemExit("Could not parse UniMTS metric line from official output.")
    metric_values = {
        "accuracy": float(match.group(1)),
        "precision_macro": float(match.group(2)),
        "recall_macro": float(match.group(3)),
        "f1_macro": float(match.group(4)),
    }
    if match.group(5) is not None:
        metric_values["f1_weighted"] = float(match.group(5))
    return {
        **metric_values,
        "runtime_seconds": round(time.time() - start, 3),
        "official_stdout_tail": lines[-80:],
        "command": cmd,
        "split": split,
    }


def metric_summary(metric_rows: list[dict]) -> dict:
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    if len(metric_rows) == 1:
        return {key: float(metric_rows[0][key]) for key in keys if key in metric_rows[0]}
    present_keys = [key for key in keys if all(key in row for row in metric_rows)]
    return summarize_fold_metrics([{key: row[key] for key in present_keys} for row in metric_rows])


def make_entry(benchmark_id: str, metrics: dict, fold_rows: list[dict], meta: dict, args) -> list[dict]:
    bench = load_benchmark(benchmark_id)
    primary = bench["metrics"]["primary"]
    metric_values = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
    }
    if "f1_weighted" in metrics:
        metric_values["f1_weighted"] = metrics["f1_weighted"]
    if primary not in metric_values:
        if primary == "f1_weighted":
            raise SystemExit(
                f"{benchmark_id} primary metric is f1_weighted, but the official UniMTS script output did not include it."
            )
        raise SystemExit(f"{benchmark_id} primary metric {primary} is unavailable from UniMTS official output.")

    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metric_values[primary]),
        "primary_metric": primary,
        "metrics": metric_values,
        "source": "official_code_adapter",
        "protocol_status": "canonical",
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": OFFICIAL_COMMIT,
        "official_model_script": "finetune_custom.py",
        "official_configuration_source": "UniMTS README full-shot custom fine-tuning command",
        "official_adapter_note": "Only converts leaderboard data into the official UniMTS custom npy/json format; training/evaluation uses upstream finetune_custom.py.",
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gpu": args.gpu,
        "input_adapter": meta,
        "runtime_seconds": metrics["runtime_seconds"],
        "official_stdout_tail": metrics["official_stdout_tail"],
    }
    if len(fold_rows) > 1:
        result["folds"] = len(fold_rows)
        result["fold_metrics"] = fold_rows
    ref = PAPER_REFERENCE.get(benchmark_id)
    if ref:
        ref_metric = ref["metric"]
        observed = metric_values.get(ref_metric, metric_values[primary])
        gap = float(observed) - float(ref["score"])
        directly_comparable = bool(ref.get("directly_comparable", True))
        result.update(
            {
                "paper_reference_score": float(ref["score"]),
                "paper_reference_metric": ref_metric,
                "paper_reference_source": ref["source"],
                "paper_reference_protocol_note": ref["protocol_note"],
                "paper_reference_gap_signed": gap,
                "paper_reference_gap_abs": abs(gap),
                "paper_reference_gap_threshold": GAP_THRESHOLD,
                "paper_reference_directly_comparable": directly_comparable,
            }
        )
        if directly_comparable:
            result["paper_reference_gap_status"] = "large_gap_check_protocol" if abs(gap) > GAP_THRESHOLD else "within_expected_range"
            result["calibration_status"] = "failed" if abs(gap) > GAP_THRESHOLD else "passed"
        else:
            result["paper_reference_gap_status"] = "not_directly_comparable_dataset_variant"
            result["calibration_status"] = "extension_after_uci_calibration_passed"

    return [
        {
            "id": "unimts",
            "method_name": "UniMTS",
            "year": 2024,
            "method_family": "Unified contrastive pre-training for motion time series",
            "submission_type": "foundation_model",
            "coverage_status": "canonical_official_code_adapter",
            "summary": "UniMTS official fine-tuning code run against the leaderboard's canonical HAR benchmark data.",
            "implementation": "Exports canonical data to the official custom npy/json format and invokes the upstream UniMTS finetune_custom.py full fine-tuning script.",
            "har_scope": "IMU motion time-series recognition using contrastive motion-text pretraining and ST-GCN sensor encoder.",
            "datasets": [
                {
                    "benchmark_id": benchmark_id,
                    "dataset": bench.get("dataset") or bench.get("name"),
                    "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
                    "subjects": str(bench.get("subjects", "")),
                    "device_placement": bench.get("placement", ""),
                    "reported_effect": f"Official UniMTS full-shot adapter rerun {primary} {metric_values[primary]:.4f}.",
                    "protocol_note": f"Canonical benchmark split; UniMTS custom input adapter: {meta['input_channels']} at joint_list={meta['joint_list']}.",
                }
            ],
            "results": [result],
            "sources": [
                OFFICIAL_REPO_URL,
                "https://arxiv.org/abs/2410.19818",
                bench.get("dataset_url", ""),
            ],
        }
    ]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci_har"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mode", choices=["full", "probe", "random"], default="full")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gyro", action="store_true")
    parser.add_argument("--run-tag", default="iotbench_unimts_uci_har_full")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for benchmark_id in args.datasets:
        meta = prepare_dataset(benchmark_id)
        fold_rows = []
        metric_rows = []
        total_runtime = 0.0
        stdout_tail = []
        for split in meta["splits"]:
            print(f"UniMTS official {benchmark_id} split={split['name']}", flush=True)
            row = run_official(meta, split, args)
            total_runtime += float(row["runtime_seconds"])
            stdout_tail = row["official_stdout_tail"]
            metric_rows.append(row)
            split_metrics = {
                "accuracy": row["accuracy"],
                "precision_macro": row["precision_macro"],
                "recall_macro": row["recall_macro"],
                "f1_macro": row["f1_macro"],
            }
            if "f1_weighted" in row:
                split_metrics["f1_weighted"] = row["f1_weighted"]
            fold_rows.append(
                {
                    **split,
                    "metrics": split_metrics,
                    "runtime_seconds": row["runtime_seconds"],
                }
            )
        metrics = metric_summary(metric_rows)
        metrics["runtime_seconds"] = round(total_runtime, 3)
        metrics["official_stdout_tail"] = stdout_tail
        out_dir = RUNS_DIR / f"{benchmark_id}_unimts_official"
        if args.clean_output and out_dir.exists():
            shutil.rmtree(out_dir)
        write_json(out_dir / "results.json", make_entry(benchmark_id, metrics, fold_rows, meta, args))
        primary = load_benchmark(benchmark_id)["metrics"]["primary"]
        print(f"{benchmark_id} / UniMTS official: {primary}={metrics[primary]:.4f}, runtime={metrics['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
