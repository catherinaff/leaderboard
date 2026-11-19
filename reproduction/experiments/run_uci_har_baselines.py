"""Run canonical UCI-HAR classical baselines.

The UCI-HAR benchmark is already pre-segmented by its authors. This script uses
the fixed train/test split from the dataset and benchmark JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "datasets" / "uci_har" / "UCI HAR Dataset"
RUN_DIR = ROOT / "experiments" / "runs" / "uci_har_baselines"


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = DATA_DIR / name
    x = np.loadtxt(split_dir / f"X_{name}.txt", dtype=np.float32)
    y = np.loadtxt(split_dir / f"y_{name}.txt", dtype=np.int64)
    return x, y


def evaluate(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    start = time.time()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "id": f"{name.lower().replace(' ', '_')}_uci_har",
        "method_name": name,
        "submission_type": "platform",
        "coverage_status": "canonical_rerun",
        "summary": f"{name} trained on the canonical UCI-HAR fixed train split and evaluated on the canonical fixed test split.",
        "implementation": "scikit-learn baseline over the official 561-dimensional UCI-HAR feature vectors released by the dataset authors.",
        "har_scope": "Classical feature-vector HAR baseline for the pre-segmented UCI-HAR smartphone benchmark.",
        "datasets": [
            {
                "benchmark_id": "uci_har",
                "dataset": "UCI-HAR",
                "activity_task": "6-class activity recognition: walking, upstairs, downstairs, sitting, standing, laying.",
                "subjects": "30 total; official split train subjects 1-21 and test subjects 22-30.",
                "device_placement": "Waist-mounted Samsung Galaxy S II smartphone accelerometer and gyroscope.",
                "reported_effect": "",
                "protocol_note": "Canonical local rerun using the dataset authors' fixed feature train/test files."
            }
        ],
        "results": [
            {
                "benchmark_id": "uci_har",
                "primary_score": float(accuracy_score(y_test, pred)),
                "metrics": {
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1_macro": float(f1_score(y_test, pred, average="macro")),
                    "f1_weighted": float(f1_score(y_test, pred, average="weighted"))
                },
                "source": "local_canonical_rerun",
                "protocol_status": "canonical",
                "runtime_seconds": round(time.time() - start, 3)
            }
        ],
        "sources": [
            "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
            "https://scikit-learn.org/stable/"
        ]
    }


def main() -> int:
    if not (DATA_DIR / "train" / "X_train.txt").is_file():
        raise SystemExit("UCI-HAR data missing. Run experiments/download_uci_har.py first.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")

    models = [
        ("Linear SVM", make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=42))),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")),
        ("kNN", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))),
    ]

    results = [evaluate(name, model, x_train, y_train, x_test, y_test) for name, model in models]
    out = RUN_DIR / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    for item in results:
        score = item["results"][0]["metrics"]["accuracy"]
        f1 = item["results"][0]["metrics"]["f1_macro"]
        print(f"{item['method_name']}: accuracy={score:.4f}, f1_macro={f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

