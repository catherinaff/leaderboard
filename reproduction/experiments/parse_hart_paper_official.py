"""Parse official HART paper-protocol runs into IoTBench result JSON."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
OFFICIAL_REPO_URL = "https://github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices"

PAPER_REFERENCE = {
    "HART": {
        "method_id": "hart",
        "method_name": "HART",
        "summary": "HART paper-protocol reproduction parsed from the official TensorFlow repository output.",
        "implementation": "Runs the official HART main.py pipeline and records the official GlobalACC.csv/log metrics.",
        "datasets": {
            "uci": {
                "benchmark_id": "uci_har",
                "official_dataset": "UCI",
                "primary_metric": "accuracy",
                "score": 0.9449,
                "source": "official HART README result table",
            },
            "motionsense": {
                "benchmark_id": "motionsense",
                "official_dataset": "MotionSense",
                "primary_metric": "f1_score",
                "score": 0.9816,
                "source": "HART/MobileHART paper result table",
            },
            "hhar": {
                "benchmark_id": "hhar",
                "official_dataset": "HHAR",
                "primary_metric": "f1_macro",
                "score": 0.8329,
                "source": "HART paper Table 5 unseen-device evaluation",
            },
        },
    },
    "MobileHART": {
        "method_id": "mobilehart",
        "method_name": "MobileHART",
        "summary": "MobileHART paper-protocol reproduction parsed from the official TensorFlow repository output.",
        "implementation": "Runs the official MobileHART main.py pipeline and records the official GlobalACC.csv/log metrics.",
        "datasets": {
            "uci": {
                "benchmark_id": "uci_har",
                "official_dataset": "UCI",
                "primary_metric": "accuracy",
                "score": 0.9720,
                "source": "official HART/MobileHART README result table",
            },
            "motionsense": {
                "benchmark_id": "motionsense",
                "official_dataset": "MotionSense",
                "primary_metric": "f1_score",
                "score": 0.9845,
                "source": "official HART/MobileHART README result table",
            },
            "hhar": {
                "benchmark_id": "hhar",
                "official_dataset": "HHAR",
                "primary_metric": "f1_macro",
                "score": 0.9819,
                "source": "official HART/MobileHART README result table",
            },
        },
    },
}

GAP_THRESHOLD = 0.05


def read_accuracy(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Test accuracy:\s*([0-9.]+)%", text)
    if not matches:
        return None
    return float(matches[-1]) / 100.0


def read_global_csv(csv_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not csv_path.is_file():
        return metrics
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            key = row[0].strip().lower().replace("\n", " ")
            try:
                value = float(row[1])
            except ValueError:
                continue
            if value > 1.0:
                value = value / 100.0
            if "test weighted f1" in key:
                metrics["f1_weighted"] = value
            elif "test micro f1" in key:
                metrics["f1_micro"] = value
            elif "test macro f1" in key:
                metrics["f1_macro"] = value
            elif key == "train:":
                metrics["train_accuracy"] = value
            elif key == "validation:":
                metrics["validation_accuracy"] = value
    return metrics


def official_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted({d for cfg in PAPER_REFERENCE.values() for d in cfg["datasets"]}))
    parser.add_argument("--architecture", default="HART", choices=sorted(PAPER_REFERENCE))
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--global-csv", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-tag", default="hart_paper_official")
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    method_cfg = PAPER_REFERENCE[args.architecture]
    ref = method_cfg["datasets"][args.dataset]
    global_csv = args.global_csv
    if global_csv is None:
        global_csv = (
            args.official_repo
            / "HART_Results"
            / f"{args.architecture}_16frameLength_16TimeStep_192ProjectionSize_0.005LR"
            / ref["official_dataset"]
            / "GlobalACC.csv"
        )

    metrics = read_global_csv(global_csv)
    accuracy = read_accuracy(args.log)
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    if "f1_macro" in metrics:
        metrics["f1_score"] = metrics["f1_macro"]

    primary_metric = ref["primary_metric"]
    if primary_metric not in metrics:
        raise SystemExit(f"missing primary metric {primary_metric}; parsed metrics={metrics}")

    score = float(metrics[primary_metric])
    gap = score - float(ref["score"])
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": score,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "source": "official_code_paper_protocol",
        "protocol_status": "paper_protocol",
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": official_commit(args.official_repo),
        "official_pipeline": f"DATA_* preprocessing -> main.py --architecture {args.architecture}",
        "official_dataset": ref["official_dataset"],
        "official_result_csv": str(global_csv),
        "official_log": str(args.log),
        "architecture": args.architecture,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.005,
        "label_smoothing": 0.1,
        "projection_dim": 192,
        "frame_length": 16,
        "time_step": 16,
        "paper_reference_score": float(ref["score"]),
        "paper_reference_metric": primary_metric,
        "paper_reference_source": ref["source"],
        "paper_reference_gap_signed": gap,
        "paper_reference_gap_abs": abs(gap),
        "paper_reference_gap_threshold": GAP_THRESHOLD,
        "paper_reference_gap_status": "large_gap_check_protocol" if abs(gap) > GAP_THRESHOLD else "within_expected_range",
        "calibration_status": "failed" if abs(gap) > GAP_THRESHOLD else "passed",
        "parsed_at_unix": time.time(),
        "run_tag": args.run_tag,
    }
    entry = {
        "id": method_cfg["method_id"],
        "method_name": method_cfg["method_name"],
        "year": 2023,
        "method_family": "Lightweight transformer for IMU",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_official_code",
        "summary": method_cfg["summary"],
        "implementation": method_cfg["implementation"],
        "har_scope": "Transformer-based accelerometer/gyroscope HAR on mobile/wearable datasets.",
        "datasets": [],
        "results": [result],
        "sources": [OFFICIAL_REPO_URL],
    }
    out_path = RUNS_DIR / f"{ref['benchmark_id']}_{method_cfg['method_id']}_paper_official" / "results.json"
    write_json(out_path, [entry])
    print(f"wrote {out_path}: {primary_metric}={score:.4f}, gap={abs(gap):.4f}")

    if args.merge:
        subprocess.check_call([sys.executable, "scripts/merge_experiment_results.py", str(out_path)], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
