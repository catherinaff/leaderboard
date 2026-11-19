"""Run a fast linear hinge baseline for larger LOSO HAR tracks.

This is intentionally separate from run_classical_baselines.py. The regular
LinearSVC baseline repeatedly timed out on OPPORTUNITY and RealWorld-HAR LOSO
folds, so this runner uses train-fold feature selection plus
SGDClassifier(loss="hinge").
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from har_utils import RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, window_features, write_json


BENCHMARKS = ("opportunity_a", "opportunity_b", "realworld_har")
SOURCE = "local_fast_linear_svm_rerun"


def make_model(seed: int, k_features: int, alpha: float, max_iter: int, tol: float):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=k_features),
        SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )


def evaluate_loso(x_feat: np.ndarray, y: np.ndarray, subjects: np.ndarray, args) -> tuple[dict, list[dict]]:
    fold_rows = []
    metric_rows = []
    k_features = min(args.k_features, x_feat.shape[1])
    for subject in sorted(set(int(s) for s in subjects)):
        test_idx = subjects == subject
        train_idx = ~test_idx
        model = make_model(args.seed, k_features, args.alpha, args.max_iter, args.tol)
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
    return summarize_fold_metrics(metric_rows), fold_rows


def dataset_detail(bench: dict, metrics: dict, args) -> dict:
    primary = bench["metrics"]["primary"]
    protocol = bench.get("protocol", {})
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Local canonical fast linear hinge rerun {primary} {metrics[primary]:.4f}.",
        "protocol_note": (
            f"Canonical {protocol.get('type', 'protocol')} with {protocol.get('window_size_samples')} sample windows "
            f"and {protocol.get('stride_samples')} sample stride. Implementation uses SelectKBest(k={min(args.k_features, int(args.feature_count))}) "
            "inside each training fold plus SGDClassifier(loss='hinge'), not scikit-learn LinearSVC."
        ),
    }


def make_entry(benchmark_id: str, metrics: dict, fold_rows: list[dict], runtime: float, args) -> dict:
    bench = load_benchmark(benchmark_id)
    primary = bench["metrics"]["primary"]
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "source": SOURCE,
        "protocol_status": "canonical",
        "runtime_seconds": round(runtime, 3),
        "seed": args.seed,
        "folds": len(fold_rows),
        "fold_metrics": fold_rows,
        "feature_selector": "SelectKBest(f_classif)",
        "selected_features": min(args.k_features, int(args.feature_count)),
        "optimizer": "SGDClassifier(loss='hinge')",
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    return {
        "id": f"linear_svm_fast_{benchmark_id}",
        "method_name": "Linear SVM (fast SGD hinge)",
        "year": "",
        "method_family": "Classical linear classifier",
        "submission_type": "platform",
        "coverage_status": "canonical_fast_linear_rerun",
        "summary": (
            f"Fast linear hinge baseline for the canonical {bench['name']} protocol, added because the "
            "standard LinearSVC runner timed out on larger LOSO feature matrices."
        ),
        "implementation": (
            "scikit-learn pipeline over deterministic statistical window features: StandardScaler, "
            "fold-local SelectKBest(f_classif), then SGDClassifier(loss='hinge', class_weight='balanced'). "
            "This is a fast linear SVM-style baseline, not the original LinearSVC implementation used for the other classical rows."
        ),
        "har_scope": "Classical HAR baseline over fixed-window statistical features for larger LOSO HAR datasets.",
        "datasets": [dataset_detail(bench, metrics, args)],
        "results": [result],
        "sources": [
            bench.get("dataset_url", ""),
            "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html",
        ],
    }


def run_one(benchmark_id: str, args) -> None:
    bench = load_benchmark(benchmark_id)
    x, y, subjects, _meta = load_windows(benchmark_id)
    x_feat = window_features(x)
    args.feature_count = x_feat.shape[1]
    start = time.time()
    metrics, fold_rows = evaluate_loso(x_feat, y, subjects, args)
    runtime = time.time() - start
    out_dir = RUNS_DIR / f"{benchmark_id}_linear_svm_fast"
    write_json(out_dir / "results.json", [make_entry(benchmark_id, metrics, fold_rows, runtime, args)])
    primary = bench["metrics"]["primary"]
    print(f"{benchmark_id} / Linear SVM (fast SGD hinge): {primary}={metrics[primary]:.4f}, runtime={runtime:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-features", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    for benchmark_id in args.datasets:
        run_one(benchmark_id, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
