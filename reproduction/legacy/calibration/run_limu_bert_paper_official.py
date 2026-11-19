"""Run LIMU-BERT exactly through the official paper pipeline.

The existing canonical adapter is useful for leaderboard splits, but the LIMU-BERT
paper's LIMU-GRU numbers are produced by the official pipeline:

1. use the repository's preprocessed data_20_120.npy / label_20_120.npy,
2. load the released pre-trained LIMU-BERT checkpoint,
3. generate embeddings with embedding.py,
4. train classifier.py with 80% unlabeled split and 1% balanced labeled data.

This wrapper executes those official scripts without replacing their model or
split code, parses the reported test metrics, and writes a mergeable result row.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "LIMU-BERT-Public"
OFFICIAL_REPO_URL = "https://github.com/dapowan/LIMU-BERT-Public"
OFFICIAL_COMMIT_FALLBACK = "decffee7ecb4e2e5d7244b1a759cb80b752dc6c2"
PYTHON = "/opt/conda/envs/iotbench-har/bin/python"
PAPER_REFERENCE = {
    "uci": {
        "benchmark_id": "uci_har",
        "label_index": 0,
        "primary": "accuracy",
        "score": 0.924,
        "metrics": {"accuracy": 0.924, "f1_macro": 0.923},
    },
    "hhar": {
        "benchmark_id": "hhar",
        "label_index": 2,
        "primary": "accuracy",
        "score": 0.964,
        "metrics": {"accuracy": 0.964, "f1_macro": 0.962},
    },
}
GAP_THRESHOLD = 0.05


def official_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=OFFICIAL_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return OFFICIAL_COMMIT_FALLBACK


def run_cmd(cmd: list[str], log_lines: list[str], env: dict[str, str]) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=OFFICIAL_REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line.rstrip())
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"Command failed with exit {code}: {' '.join(cmd)}")


def make_train_config(epochs: int) -> Path:
    base = json.loads((OFFICIAL_REPO / "config" / "train.json").read_text())
    base["n_epochs"] = int(epochs)
    base["save_steps"] = 100
    out = OFFICIAL_REPO / "config" / f"train_iotbench_{epochs}.json"
    out.write_text(json.dumps(base, indent=2) + "\n")
    return out


def parse_best(lines: list[str]) -> dict:
    best = next((line for line in reversed(lines) if line.startswith("Best Accuracy:")), "")
    match = re.search(
        r"Best Accuracy:\s*([0-9.]+)/([0-9.]+)/([0-9.]+),\s*F1:\s*([0-9.]+)/([0-9.]+)/([0-9.]+)",
        best,
    )
    if not match:
        raise SystemExit("Could not parse official Best Accuracy/F1 line.")
    return {
        "train_accuracy": float(match.group(1)),
        "validation_accuracy": float(match.group(2)),
        "accuracy": float(match.group(3)),
        "train_f1_macro": float(match.group(4)),
        "validation_f1_macro": float(match.group(5)),
        "f1_macro": float(match.group(6)),
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_dataset(dataset: str, args) -> None:
    if dataset not in PAPER_REFERENCE:
        raise SystemExit(f"Unsupported LIMU-BERT paper dataset: {dataset}")
    ref = PAPER_REFERENCE[dataset]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train_cfg = make_train_config(args.epochs)
    (OFFICIAL_REPO / "embed").mkdir(exist_ok=True)
    logs: list[str] = []
    start = time.time()

    embed_path = OFFICIAL_REPO / "embed" / f"embed_{dataset}_{dataset}_20_120.npy"
    if args.force_embedding or not embed_path.is_file():
        run_cmd([PYTHON, "embedding.py", "v1", dataset, "20_120", "-f", dataset, "-g", "0"], logs, env)

    save_name = f"iotbench_{dataset}_gru_paper_{args.epochs}"
    run_cmd(
        [
            PYTHON,
            "classifier.py",
            "v2",
            dataset,
            "20_120",
            "-f",
            dataset,
            "-s",
            save_name,
            "-l",
            str(ref["label_index"]),
            "-g",
            "0",
            "-t",
            str(train_cfg),
        ],
        logs,
        env,
    )
    metrics = parse_best(logs)
    primary = ref["primary"]
    gap = float(metrics[primary]) - float(ref["score"])
    runtime = round(time.time() - start, 3)
    out_dir = RUNS_DIR / f"{ref['benchmark_id']}_limu_bert_paper_official"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": float(metrics[primary]),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "train_accuracy": metrics["train_accuracy"],
            "validation_accuracy": metrics["validation_accuracy"],
            "train_f1_macro": metrics["train_f1_macro"],
            "validation_f1_macro": metrics["validation_f1_macro"],
        },
        "source": "official_code_paper_protocol",
        "protocol_status": "paper_protocol",
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": official_commit(),
        "official_pipeline": "embedding.py -> classifier.py",
        "official_model_version_embedding": "v1",
        "official_model_version_classifier": "v2",
        "official_dataset": f"dataset/{dataset}/data_20_120.npy and label_20_120.npy",
        "official_pretrain_checkpoint": f"saved/pretrain_base_{dataset}_20_120/{dataset}.pt",
        "official_classifier": "classifier.py v2/base_gru",
        "official_classifier_config": "config/classifier.json: gru_v2",
        "official_train_config": f"config/train_iotbench_{args.epochs}.json derived from config/train.json",
        "official_training_rate": 0.8,
        "official_validation_rate": 0.1,
        "official_label_rate": 0.01,
        "official_balance": True,
        "official_train_seed": 3431,
        "label_index": ref["label_index"],
        "epochs": args.epochs,
        "runtime_seconds": runtime,
        "paper_reference_score": float(ref["score"]),
        "paper_reference_metric": primary,
        "paper_reference_metrics": ref["metrics"],
        "paper_reference_source": "LIMU-BERT paper Table 2, 1% labeled data, LIMU-GRU",
        "paper_reference_gap_signed": gap,
        "paper_reference_gap_abs": abs(gap),
        "paper_reference_gap_threshold": GAP_THRESHOLD,
        "paper_reference_gap_status": "large_gap_check_protocol" if abs(gap) > GAP_THRESHOLD else "within_expected_range",
        "calibration_status": "failed" if abs(gap) > GAP_THRESHOLD else "passed",
        "official_stdout_tail": logs[-120:],
    }
    entry = {
        "id": "limu_bert",
        "method_name": "LIMU-BERT",
        "year": 2021,
        "method_family": "Self-supervised BERT for IMU",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_official_code",
        "summary": "LIMU-BERT paper-protocol reproduction using the official embedding.py + classifier.py pipeline.",
        "implementation": "Executes the official scripts against the repository's preprocessed 20_120 dataset and released pretrain checkpoint.",
        "har_scope": "Transformer representation learning for 6-axis IMU activity recognition with a GRU classifier head.",
        "datasets": [],
        "results": [result],
        "sources": [OFFICIAL_REPO_URL, "https://dl.acm.org/doi/10.1145/3485730.3485937"],
    }
    write_json(out_dir / "results.json", [entry])
    print(f"{dataset} LIMU-BERT paper official: {primary}={metrics[primary]:.4f}, gap={abs(gap):.4f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hhar"], choices=sorted(PAPER_REFERENCE))
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-embedding", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        run_dataset(dataset, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
