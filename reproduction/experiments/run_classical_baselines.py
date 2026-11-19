"""Run canonical classical HAR baselines on processed window datasets."""
from __future__ import annotations

import argparse
import time
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from har_utils import RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, window_features, write_json


METHODS = {
    "linear_svm": {
        "name": "Linear SVM",
        "family": "Classical linear classifier",
        "model": lambda seed, trees: make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=seed)),
        "implementation": "scikit-learn LinearSVC over deterministic statistical features extracted from canonical sliding windows.",
    },
    "random_forest": {
        "name": "Random Forest",
        "family": "Classical ensemble",
        "model": lambda seed, trees: RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1, class_weight="balanced"),
        "implementation": "scikit-learn RandomForestClassifier over deterministic statistical features extracted from canonical sliding windows.",
    },
    "knn": {
        "name": "kNN",
        "family": "Classical instance-based classifier",
        "model": lambda seed, trees: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "implementation": "scikit-learn 5-nearest-neighbor classifier over deterministic statistical features extracted from canonical sliding windows.",
    },
}


def fixed_subject_split(subjects: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    train_subjects = set(int(s) for s in meta.get("train_subjects", []))
    test_subjects = set(int(s) for s in meta.get("test_subjects", []))
    if not train_subjects or not test_subjects:
        raise SystemExit("Fixed split dataset requires train_subjects and test_subjects in processed metadata.")
    train_idx = np.asarray([int(s) in train_subjects for s in subjects])
    test_idx = np.asarray([int(s) in test_subjects for s in subjects])
    if not train_idx.any() or not test_idx.any():
        raise SystemExit("Fixed split produced empty train or test partition.")
    return train_idx, test_idx


def evaluate_fixed(model, x_feat, y, subjects, meta) -> tuple[dict, list[dict]]:
    train_idx, test_idx = fixed_subject_split(subjects, meta)
    model.fit(x_feat[train_idx], y[train_idx])
    pred = model.predict(x_feat[test_idx])
    return metric_bundle(y[test_idx], pred), [
        {
            "split": "fixed_subject_split",
            "train_windows": int(train_idx.sum()),
            "test_windows": int(test_idx.sum()),
            "metrics": metric_bundle(y[test_idx], pred),
        }
    ]


def evaluate_loso(model_factory, x_feat, y, subjects, seed, trees) -> tuple[dict, list[dict]]:
    fold_rows = []
    metric_rows = []
    for subject in sorted(set(int(s) for s in subjects)):
        test_idx = subjects == subject
        train_idx = ~test_idx
        if not train_idx.any() or not test_idx.any():
            continue
        model = model_factory(seed, trees)
        model.fit(x_feat[train_idx], y[train_idx])
        pred = model.predict(x_feat[test_idx])
        metrics = metric_bundle(y[test_idx], pred)
        metric_rows.append(metrics)
        fold_rows.append(
            {
                "test_subject": int(subject),
                "train_windows": int(train_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
            }
        )
    if not metric_rows:
        raise SystemExit("LOSO evaluation produced no folds.")
    return summarize_fold_metrics(metric_rows), fold_rows


def dataset_detail(bench: dict, result_metrics: dict, primary: str) -> dict:
    protocol = bench.get("protocol", {})
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Local canonical rerun {primary} {result_metrics[primary]:.4f}.",
        "protocol_note": f"{protocol.get('type', 'Canonical')} with {protocol.get('window_size_samples')} sample windows and {protocol.get('stride_samples')} sample stride.",
    }


def make_entry(benchmark_id: str, method_key: str, metrics: dict, fold_rows: list[dict], runtime: float, args) -> dict:
    bench = load_benchmark(benchmark_id)
    spec = METHODS[method_key]
    primary = bench["metrics"]["primary"]
    source = "local_canonical_rerun"
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "source": source,
        "protocol_status": "canonical",
        "runtime_seconds": round(runtime, 3),
        "seed": args.seed,
    }
    if "Leave-One-Subject-Out" in bench.get("protocol", {}).get("type", ""):
        result["folds"] = len(fold_rows)
        result["fold_metrics"] = fold_rows

    return {
        "id": f"{method_key}_{benchmark_id}",
        "method_name": spec["name"],
        "year": "",
        "method_family": spec["family"],
        "submission_type": "platform",
        "coverage_status": "canonical_rerun",
        "summary": f"{spec['name']} trained and evaluated locally on the canonical {bench['name']} protocol.",
        "implementation": spec["implementation"],
        "har_scope": "Classical HAR baseline over fixed-window statistical features.",
        "datasets": [dataset_detail(bench, metrics, primary)],
        "results": [result],
        "sources": [
            bench.get("dataset_url", ""),
            "https://scikit-learn.org/stable/",
        ],
    }


def run_one(benchmark_id: str, method_key: str, args) -> None:
    bench = load_benchmark(benchmark_id)
    x, y, subjects, meta = load_windows(benchmark_id)
    x_feat = window_features(x)
    spec = METHODS[method_key]
    start = time.time()
    protocol_type = bench.get("protocol", {}).get("type", "")
    if "Leave-One-Subject-Out" in protocol_type:
        metrics, fold_rows = evaluate_loso(spec["model"], x_feat, y, subjects, args.seed, args.rf_trees)
    else:
        model = spec["model"](args.seed, args.rf_trees)
        metrics, fold_rows = evaluate_fixed(model, x_feat, y, subjects, meta)
    runtime = time.time() - start
    entry = make_entry(benchmark_id, method_key, metrics, fold_rows, runtime, args)
    out_dir = RUNS_DIR / f"{benchmark_id}_{method_key}"
    write_json(out_dir / "results.json", [entry])
    primary = bench["metrics"]["primary"]
    print(f"{benchmark_id} / {spec['name']}: {primary}={metrics[primary]:.4f}, runtime={runtime:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=300)
    args = parser.parse_args(argv)

    for benchmark_id in args.datasets:
        for method_key in args.methods:
            run_one(benchmark_id, method_key, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
