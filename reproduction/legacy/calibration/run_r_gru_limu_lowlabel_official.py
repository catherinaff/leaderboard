"""Run R-GRU under the LIMU-BERT 1% labeled-data benchmark protocol.

This uses the recurrent GRU baseline shipped in the LIMU-BERT public
repository (`benchmark.py` + `models.ClassifierGRU`, config `gru_v1`). It
reproduces the R-GRU row in LIMU-BERT Table 2: official 20_120 arrays, random
80/10/10 split, and a balanced 1% labeled training subset.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "LIMU-BERT-Public"
OFFICIAL_REPO_URL = "https://github.com/dapowan/LIMU-BERT-Public"
OFFICIAL_COMMIT_FALLBACK = "decffee7ecb4e2e5d7244b1a759cb80b752dc6c2"
GAP_THRESHOLD = 0.05


class Tee(StringIO):
    def __init__(self, stream) -> None:
        super().__init__()
        self.stream = stream

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.stream.flush()
        return super().write(text)

    def flush(self) -> None:
        self.stream.flush()
        super().flush()


PAPER_REFERENCE = {
    "hhar": {
        "benchmark_id": "hhar",
        "label_index": 2,
        "metrics": {"accuracy": 0.849, "f1_macro": 0.832},
    },
    "uci": {
        "benchmark_id": "uci_har",
        "label_index": 0,
        "metrics": {"accuracy": 0.760, "f1_macro": 0.741},
    },
    "motion": {
        "benchmark_id": "motionsense",
        "label_index": 0,
        "metrics": {"accuracy": 0.846, "f1_macro": 0.806},
    },
    "shoaib": {
        "benchmark_id": "shoaib",
        "label_index": 0,
        "metrics": {"accuracy": 0.785, "f1_macro": 0.787},
    },
}


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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_train_config(epochs: int) -> Path:
    cfg = json.loads((OFFICIAL_REPO / "config" / "train.json").read_text(encoding="utf-8"))
    cfg["n_epochs"] = int(epochs)
    cfg["save_steps"] = 100
    out = OFFICIAL_REPO / "config" / f"train_iotbench_r_gru_lowlabel_{epochs}.json"
    write_json(out, cfg)
    return out


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_summary(y_true: np.ndarray, logits: np.ndarray, class_count: int) -> dict[str, float]:
    y_pred = np.argmax(logits, axis=1)
    accuracy = float(np.mean(y_true == y_pred))
    f1s = []
    supports = []
    for cls in range(class_count):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        support = float(np.sum(y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        supports.append(support)
    f1_arr = np.asarray(f1s, dtype=np.float64)
    supports_arr = np.asarray(supports, dtype=np.float64)
    return {
        "accuracy": accuracy,
        "f1_macro": float(f1_arr.mean()),
        "f1_weighted": float((f1_arr * supports_arr).sum() / supports_arr.sum()),
        "f1_micro": accuracy,
    }


def load_official_modules():
    if not OFFICIAL_REPO.is_dir():
        raise SystemExit(f"Missing LIMU-BERT official repo: {OFFICIAL_REPO}")
    if "seaborn" not in sys.modules:
        try:
            __import__("seaborn")
        except ModuleNotFoundError:
            stub = ModuleType("seaborn")
            stub.heatmap = lambda *args, **kwargs: None
            sys.modules["seaborn"] = stub
    if "matplotlib.pyplot" not in sys.modules:
        try:
            __import__("matplotlib.pyplot")
        except ModuleNotFoundError:
            mpl = ModuleType("matplotlib")
            pyplot = ModuleType("matplotlib.pyplot")
            for name in ("figure", "show", "savefig", "close", "xlabel", "ylabel", "title", "tight_layout"):
                setattr(pyplot, name, lambda *args, **kwargs: None)
            mpl.pyplot = pyplot
            sys.modules["matplotlib"] = mpl
            sys.modules["matplotlib.pyplot"] = pyplot
    sys.path.insert(0, str(OFFICIAL_REPO))
    from benchmark import classify_benchmark  # type: ignore
    from config import create_io_config, load_dataset_label_names, load_dataset_stats, load_model_config  # type: ignore

    return classify_benchmark, create_io_config, load_dataset_label_names, load_dataset_stats, load_model_config


def calibration_status(gaps: dict[str, float], primary: str) -> tuple[str, str]:
    max_abs_gap = max(abs(value) for value in gaps.values())
    if max_abs_gap <= GAP_THRESHOLD:
        return "within_expected_range", "passed"
    if abs(gaps.get(primary, 999.0)) <= GAP_THRESHOLD:
        return "primary_within_range_secondary_metric_borderline", "passed_primary_metric_secondary_metric_borderline"
    return "large_gap_check_protocol", "failed"


def run_dataset(dataset: str, args, modules) -> dict:
    if dataset not in PAPER_REFERENCE:
        raise SystemExit(f"Unsupported dataset: {dataset}")

    classify_benchmark, create_io_config, load_dataset_label_names, load_dataset_stats, load_model_config = modules
    ref = PAPER_REFERENCE[dataset]
    train_cfg = make_train_config(args.epochs)
    save_name = f"iotbench_r_gru_limu_lowlabel_{dataset}_{args.epochs}"
    old_cwd = Path.cwd()
    os.chdir(OFFICIAL_REPO)
    try:
        model_cfg = load_model_config("bench_gru", "gru", "v1")
        dataset_cfg = load_dataset_stats(dataset, "20_120")
        if model_cfg is None or dataset_cfg is None:
            raise SystemExit(f"Missing LIMU-BERT config for {dataset}")
        label_names, label_num = load_dataset_label_names(dataset_cfg, ref["label_index"])
        run_args = SimpleNamespace(
            model_version="v1",
            dataset=dataset,
            dataset_version="20_120",
            gpu=str(args.gpu) if args.gpu is not None else None,
            model_file=None,
            train_cfg=str(train_cfg),
            mask_cfg="./config/mask.json",
            label_index=int(ref["label_index"]),
            save_model=save_name,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
        )
        run_args = create_io_config(
            run_args,
            dataset,
            "20_120",
            pretrain_model=None,
            target="bench_gru",
        )

        set_all_seeds(args.seed)
        started = time.time()
        log_buffer = Tee(sys.stdout)
        with redirect_stdout(log_buffer):
            label_test, logits = classify_benchmark(
                run_args,
                int(ref["label_index"]),
                training_rate=0.8,
                label_rate=0.01,
                balance=True,
                method="gru",
            )
        elapsed = round(time.time() - started, 3)
    finally:
        os.chdir(old_cwd)

    logs = log_buffer.getvalue().splitlines()
    metrics = metric_summary(np.asarray(label_test), np.asarray(logits), int(label_num))
    gaps = {
        key: float(metrics[key]) - float(value)
        for key, value in ref["metrics"].items()
        if key in metrics
    }
    primary = "accuracy"
    gap_status, cal_status = calibration_status(gaps, primary)
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": metrics[primary],
        "metrics": metrics,
        "source": "limu_bert_official_benchmark_r_gru_lowlabel",
        "protocol_status": "paper_protocol",
        "official_code_used": True,
        "official_code_scope": "LIMU-BERT official benchmark baseline implementation of R-GRU.",
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": official_commit(),
        "official_pipeline": "benchmark.py classify_benchmark(method='gru')",
        "official_model_code": "models.ClassifierGRU",
        "official_model_config": "config/classifier.json: gru_v1",
        "official_dataset": f"dataset/{dataset}/data_20_120.npy and label_20_120.npy",
        "official_training_rate": 0.8,
        "official_validation_rate": 0.1,
        "official_label_rate": 0.01,
        "official_balance": True,
        "official_train_seed": args.seed,
        "label_index": int(ref["label_index"]),
        "label_names": label_names,
        "epochs": args.epochs,
        "runtime_seconds": elapsed,
        "paper_reference_metric": primary,
        "paper_reference_score": ref["metrics"][primary],
        "paper_reference_metrics": ref["metrics"],
        "paper_reference_source": "LIMU-BERT paper Table 2, Performance comparison on HAR with 1% labeled data, R-GRU row",
        "paper_reference_gap_signed": gaps[primary],
        "paper_reference_gap_abs": abs(gaps[primary]),
        "paper_reference_metric_gaps": gaps,
        "paper_reference_max_gap_abs": max(abs(value) for value in gaps.values()),
        "paper_reference_gap_threshold": GAP_THRESHOLD,
        "paper_reference_gap_status": gap_status,
        "calibration_status": cal_status,
        "protocol_note": "Reproduced under the LIMU-BERT 1% labeled-data protocol: official 20_120 arrays, 80/10/10 split, balanced 1% labeled training subset, seed 3431, 700 epochs.",
        "paper_training_setting": "LIMU-BERT Table 2: Performance comparison on HAR with 1% labeled data",
        "track": "Low-Label / Few-Shot calibration",
        "official_stdout_tail": logs[-120:],
    }
    if cal_status != "passed":
        result["review_note"] = "At least one reported metric is outside the 5-point tolerance; inspect protocol and official baseline settings before using this as passed evidence."

    out_dir = RUNS_DIR / f"{ref['benchmark_id']}_r_gru_limu_lowlabel_official"
    entry = {
        "id": "r_gru",
        "method_name": "R-GRU",
        "year": 2021,
        "method_family": "Recurrent GRU baseline",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_official_baseline_code",
        "summary": "R-GRU baseline rerun under LIMU-BERT's 1% labeled-data benchmark protocol.",
        "implementation": "Uses the LIMU-BERT public repository's benchmark.py and ClassifierGRU baseline over official 20_120 arrays with balanced 1% labeled training data.",
        "har_scope": "Recurrent baseline for IMU human activity recognition from 6-axis or 9-axis wearable/mobile sensor windows.",
        "datasets": [],
        "results": [result],
        "sources": [OFFICIAL_REPO_URL, "https://dl.acm.org/doi/10.1145/3485730.3485937"],
    }
    write_json(out_dir / "results.json", [entry])
    print(
        f"{dataset}: accuracy={metrics['accuracy']:.4f} "
        f"(paper {ref['metrics']['accuracy']:.4f}, gap {gaps['accuracy']:+.4f}); "
        f"macro-F1={metrics['f1_macro']:.4f} "
        f"(paper {ref['metrics']['f1_macro']:.4f}, gap {gaps['f1_macro']:+.4f}); "
        f"status={cal_status}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hhar"], choices=sorted(PAPER_REFERENCE))
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3431)
    args = parser.parse_args(argv)

    modules = load_official_modules()
    all_results = [run_dataset(dataset, args, modules) for dataset in args.datasets]
    out = RUNS_DIR / f"r_gru_limu_lowlabel_official_{int(time.time())}.json"
    write_json(out, all_results)
    print(f"Wrote aggregate: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
