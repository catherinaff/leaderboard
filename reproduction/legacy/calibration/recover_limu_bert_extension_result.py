"""Recover a LIMU-BERT extension result from the shared queue log."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_limu_bert_extension_official as runner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_id", choices=runner.EXTENSION_DATASETS)
    parser.add_argument("--log", type=Path, default=ROOT / "experiments/logs/limu_bert_extension_full_queue.log")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    metrics = runner.parse_best(lines)
    requested_primary = runner.PRIMARY_METRIC[args.benchmark_id]
    primary = requested_primary if requested_primary in metrics else "f1_macro"
    prepared = runner.prepare_official_dataset(args.benchmark_id, force=False)
    official_name = prepared["official_dataset_name"]

    result = {
        "benchmark_id": args.benchmark_id,
        "primary_score": float(metrics[primary]),
        "primary_metric": primary,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "train_accuracy": metrics["train_accuracy"],
            "validation_accuracy": metrics["validation_accuracy"],
            "train_f1_macro": metrics["train_f1_macro"],
            "validation_f1_macro": metrics["validation_f1_macro"],
        },
        "source": "official_code_extension",
        "protocol_status": "official_extension",
        "official_code_used": True,
        "official_repo": runner.OFFICIAL_REPO_URL,
        "official_commit": runner.official_commit(),
        "official_pipeline": "pretrain.py -> embedding.py -> classifier.py",
        "official_dataset": f"dataset/{official_name}/data_20_120.npy and label_20_120.npy",
        "official_model_version_pretrain": "v1",
        "official_model_version_classifier": "v2",
        "official_classifier": "classifier.py v2/base_gru",
        "official_classifier_config": "config/classifier.json: gru_v2",
        "official_training_rate": 0.8,
        "official_validation_rate": 0.1,
        "official_label_rate": 0.01,
        "official_balance": True,
        "official_train_seed": 3431,
        "requested_primary_metric": requested_primary,
        "primary_metric_resolution": (
            "official_classifier_reports_accuracy_and_f1_only; fell back to f1_macro"
            if requested_primary != primary
            else "official_classifier_reports_requested_metric"
        ),
        "pretrain_epochs": 3200,
        "epochs": 700,
        "calibration_status": "extension_after_uci_hhar_calibration_passed",
        "paper_reference_gap_status": "unavailable_for_same_benchmark_metric",
        "channel_adapter": prepared["channel_adapter"],
        "official_format_preparation": prepared,
        "official_stdout_tail": lines[-160:],
        "recovered_from_log": str(args.log),
        "parsed_at_unix": time.time(),
    }
    entry = {
        "id": "limu_bert",
        "method_name": "LIMU-BERT",
        "year": 2021,
        "method_family": "Self-supervised BERT for IMU",
        "submission_type": "platform",
        "coverage_status": "official_code_extension_after_paper_calibration",
        "summary": "LIMU-BERT official-code extension after UCI/HHAR paper calibration passed.",
        "implementation": "Runs the official LIMU-BERT-Public pretrain.py -> embedding.py -> classifier.py pipeline on IoTBench-HAR windows converted to the official 20_120 array format.",
        "har_scope": "Transformer representation learning for 6-axis IMU activity recognition with a GRU classifier head.",
        "datasets": [],
        "results": [result],
        "sources": [runner.OFFICIAL_REPO_URL, "https://dl.acm.org/doi/10.1145/3485730.3485937"],
    }
    out_path = ROOT / "experiments" / "runs" / f"{args.benchmark_id}_limu_bert_official_extension" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {primary}={metrics[primary]:.4f}")
    if args.merge:
        subprocess.check_call([sys.executable, "scripts/merge_experiment_results.py", str(out_path)], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
