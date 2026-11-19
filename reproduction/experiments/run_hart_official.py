"""Run HART with the official TensorFlow model code on canonical windows.

The official HART repository exposes Keras model builders but its full pipeline
only supports the authors' prepared dataset folders. This adapter keeps the
leaderboard's canonical split/window protocols and imports `model.HART` from the
official repository without modifying the model implementation.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.utils import class_weight

from har_utils import DATASETS_DIR, RUNS_DIR, load_benchmark, load_windows, metric_bundle, summarize_fold_metrics, write_json


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices"
OFFICIAL_COMMIT = "0b8b39bbf615c252b9a79a8b121b0665e071dcd4"
OFFICIAL_REPO_URL = "https://github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices"
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


def import_tensorflow():
    import tensorflow as tf  # type: ignore

    return tf


def import_official_model():
    if not OFFICIAL_REPO.is_dir():
        raise SystemExit(f"Official HART repository missing: {OFFICIAL_REPO}")
    sys.path.insert(0, str(OFFICIAL_REPO))
    import model as hart_model  # type: ignore

    return hart_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf = import_tensorflow()
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_uci_har() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data_dir = DATASETS_DIR / "uci_har" / "UCI HAR Dataset"
    if not (data_dir / "train" / "Inertial Signals").is_dir():
        raise SystemExit("UCI-HAR data missing. Run experiments/download_uci_har.py first.")

    xs = []
    ys = []
    subjects = []
    split_masks = []
    for split, is_train in (("train", True), ("test", False)):
        sig_dir = data_dir / split / "Inertial Signals"
        arrays = [np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32) for name in UCI_SIGNALS]
        x_split = np.stack(arrays, axis=2)
        y_split = np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=np.int64) - 1
        s_split = np.loadtxt(data_dir / split / f"subject_{split}.txt", dtype=np.int64)
        xs.append(x_split)
        ys.append(y_split)
        subjects.append(s_split)
        split_masks.append(np.full(len(y_split), is_train, dtype=bool))
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.int64),
        np.concatenate(subjects).astype(np.int64),
        {"train_mask": np.concatenate(split_masks), "split": "official_fixed"},
    )


def load_dataset(benchmark_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if benchmark_id == "uci_har":
        return load_uci_har()
    return load_windows(benchmark_id)


def fixed_subject_split(subjects: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    if "train_mask" in meta:
        train_idx = np.asarray(meta["train_mask"], dtype=bool)
        return train_idx, ~train_idx
    train_subjects = set(int(s) for s in meta.get("train_subjects", []))
    test_subjects = set(int(s) for s in meta.get("test_subjects", []))
    if not train_subjects or not test_subjects:
        raise SystemExit("Fixed split dataset requires train_subjects/test_subjects or train_mask metadata.")
    return np.asarray([int(s) in train_subjects for s in subjects]), np.asarray([int(s) in test_subjects for s in subjects])


def train_val_split(train_idx: np.ndarray, seed: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    train_positions = np.flatnonzero(train_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(train_positions)
    cutoff = max(1, int(len(train_positions) * ratio))
    train_only = np.zeros_like(train_idx, dtype=bool)
    val_only = np.zeros_like(train_idx, dtype=bool)
    train_only[train_positions[:cutoff]] = True
    val_only[train_positions[cutoff:]] = True
    if not val_only.any():
        val_only[train_positions[-1]] = True
        train_only[train_positions[-1]] = False
    return train_only, val_only


def adapt_channels(x: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    x = np.asarray(x, dtype=np.float32)
    channels = x.shape[2]
    if mode != "first6":
        raise SystemExit(f"Unsupported --channel-adapter {mode}")
    if channels == 6:
        return x, "native_6ch"
    if channels > 6:
        return x[:, :, :6], f"first6_of_{channels}ch"
    pad = np.zeros((x.shape[0], x.shape[1], 6 - channels), dtype=np.float32)
    return np.concatenate([x, pad], axis=2), f"zero_padded_{channels}ch_to_6ch"


def normalize_fold(x: np.ndarray, train_val_idx: np.ndarray) -> np.ndarray:
    mean = x[train_val_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_val_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def build_model(input_shape: tuple[int, int], classes: int, args):
    hart_model = import_official_model()
    if args.architecture != "HART":
        raise SystemExit("This runner is currently scoped to the HART method; run MobileHART separately.")
    return hart_model.HART(
        input_shape,
        classes,
        projection_dim=args.projection_dim,
        patchSize=args.frame_length,
        timeStep=args.time_step,
        num_heads=args.num_heads,
        filterAttentionHead=args.filter_attention_head,
        convKernels=args.conv_kernels,
        mlp_head_units=[args.mlp_head_units],
        dropout_rate=args.dropout_rate,
        useTokens=args.token_based,
    )


class EpochLogger:
    def __init__(self, tf, log_every: int):
        self.callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self._on_epoch_end)
        self.log_every = log_every

    def _on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        epoch_num = epoch + 1
        if epoch_num == 1 or epoch_num % self.log_every == 0:
            print(
                "HART official "
                f"epoch={epoch_num} "
                f"loss={logs.get('loss', float('nan')):.4f} "
                f"val_loss={logs.get('val_loss', float('nan')):.4f} "
                f"val_accuracy={logs.get('val_accuracy', float('nan')):.4f}",
                flush=True,
            )


def train_eval_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    adapter_note: str,
    args,
) -> tuple[dict, float]:
    tf = import_tensorflow()
    seed_all(args.seed)
    train_idx, val_idx = train_val_split(train_val_idx, args.seed, args.train_val_ratio)
    x_norm = normalize_fold(x, train_val_idx)
    classes = int(y.max() + 1)
    y_train = tf.one_hot(y[train_idx], classes)
    y_val = tf.one_hot(y[val_idx], classes)
    y_test = y[test_idx]

    model = build_model((x_norm.shape[1], x_norm.shape[2]), classes, args)
    optimizer = tf.keras.optimizers.Adam(args.lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )

    present_classes = np.unique(y[train_idx])
    weights = class_weight.compute_class_weight(class_weight="balanced", classes=present_classes, y=y[train_idx])
    class_weights = {int(cls): float(weight) for cls, weight in zip(present_classes, weights)}

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="hart_official_") as tmp:
        checkpoint_path = Path(tmp) / "bestValcheckpoint.weights.h5"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
            EpochLogger(tf, args.log_every).callback,
        ]
        model.fit(
            x=x_norm[train_idx],
            y=y_train,
            validation_data=(x_norm[val_idx], y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=0,
            class_weight=class_weights,
            callbacks=callbacks,
        )
        if checkpoint_path.is_file():
            model.load_weights(str(checkpoint_path))
        pred = model.predict(x_norm[test_idx], batch_size=args.batch_size, verbose=0).argmax(axis=1)
    runtime = time.time() - start
    tf.keras.backend.clear_session()
    return metric_bundle(y_test, pred), runtime


def evaluate_dataset(benchmark_id: str, args) -> tuple[dict, list[dict], float, str]:
    x_raw, y, subjects, meta = load_dataset(benchmark_id)
    x, adapter_note = adapt_channels(x_raw, args.channel_adapter)
    bench = load_benchmark(benchmark_id)
    protocol_type = bench.get("protocol", {}).get("type", "")
    if "Leave-One-Subject-Out" not in protocol_type:
        train_val_idx, test_idx = fixed_subject_split(subjects, meta)
        metrics, runtime = train_eval_fold(x, y, train_val_idx, test_idx, adapter_note, args)
        return metrics, [
            {
                "split": "fixed_subject_split",
                "train_val_windows": int(train_val_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
            }
        ], runtime, adapter_note

    fold_rows = []
    metric_rows = []
    total_runtime = 0.0
    for subject in sorted(set(int(s) for s in subjects)):
        test_idx = subjects == subject
        train_val_idx = ~test_idx
        print(f"HART official {benchmark_id} LOSO subject={subject}", flush=True)
        metrics, runtime = train_eval_fold(x, y, train_val_idx, test_idx, adapter_note, args)
        metric_rows.append(metrics)
        total_runtime += runtime
        fold_rows.append(
            {
                "test_subject": int(subject),
                "train_val_windows": int(train_val_idx.sum()),
                "test_windows": int(test_idx.sum()),
                "metrics": metrics,
                "runtime_seconds": round(runtime, 3),
            }
        )
    return summarize_fold_metrics(metric_rows), fold_rows, total_runtime, adapter_note


def dataset_detail(bench: dict, metrics: dict, adapter_note: str) -> dict:
    primary = bench["metrics"]["primary"]
    return {
        "benchmark_id": bench["id"],
        "dataset": bench.get("dataset") or bench.get("name"),
        "activity_task": f"{len(bench.get('classes', []))}-class activity recognition: {', '.join(bench.get('classes', []))}.",
        "subjects": str(bench.get("subjects", "")),
        "device_placement": bench.get("placement", ""),
        "reported_effect": f"Official HART code adapter canonical rerun {primary} {metrics[primary]:.4f}.",
        "protocol_note": f"Canonical benchmark split/window protocol; HART official 6-channel acc/gyro input adapter: {adapter_note}.",
    }


def tensorflow_metadata() -> dict:
    tf = import_tensorflow()
    metadata = {"tensorflow_version": tf.__version__}
    gpus = tf.config.list_physical_devices("GPU")
    metadata["tf_gpu_devices"] = [gpu.name for gpu in gpus]
    return metadata


def make_entry(benchmark_id: str, metrics: dict, fold_rows: list[dict], runtime: float, adapter_note: str, args) -> dict:
    bench = load_benchmark(benchmark_id)
    primary = bench["metrics"]["primary"]
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
        "metrics": metrics,
        "source": "official_code_adapter",
        "protocol_status": "canonical",
        "runtime_seconds": round(runtime, 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": OFFICIAL_COMMIT,
        "official_model_function": "model.HART",
        "official_configuration_source": "README quick-start and main.py defaults",
        "official_adapter_note": "Only the data adapter is leaderboard-specific; the HART model implementation is imported from the official TensorFlow repository.",
        "architecture": args.architecture,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "label_smoothing": args.label_smoothing,
        "projection_dim": args.projection_dim,
        "frame_length": args.frame_length,
        "time_step": args.time_step,
        "dropout_rate": args.dropout_rate,
        "conv_kernels": args.conv_kernels,
        "input_channel_adapter": adapter_note,
        "normalization": "per-fold z-score using train+validation windows",
        **tensorflow_metadata(),
    }
    if "Leave-One-Subject-Out" in bench.get("protocol", {}).get("type", ""):
        result["folds"] = len(fold_rows)
        result["fold_metrics"] = fold_rows
    return {
        "id": "hart",
        "method_name": "HART",
        "year": 2022,
        "method_family": "Sensor-wise lightweight transformer",
        "submission_type": "platform",
        "coverage_status": "canonical_official_code_adapter",
        "summary": "HART official TensorFlow model code run against the leaderboard's canonical HAR benchmark windows.",
        "implementation": "Imports HART from github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices and trains it with the official Adam + label-smoothed categorical cross-entropy pipeline.",
        "har_scope": "Transformer-based wearable HAR with sensor-wise attention for mobile and heterogeneous environments.",
        "datasets": [dataset_detail(bench, metrics, adapter_note)],
        "results": [result],
        "sources": [
            OFFICIAL_REPO_URL,
            "https://arxiv.org/pdf/2209.11750.pdf",
            bench.get("dataset_url", ""),
        ],
    }


def run_one(benchmark_id: str, args) -> None:
    seed_all(args.seed)
    metrics, fold_rows, runtime, adapter_note = evaluate_dataset(benchmark_id, args)
    out_dir = RUNS_DIR / f"{benchmark_id}_hart_official"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    write_json(out_dir / "results.json", [make_entry(benchmark_id, metrics, fold_rows, runtime, adapter_note, args)])
    primary = load_benchmark(benchmark_id)["metrics"]["primary"]
    print(f"{benchmark_id} / HART official: {primary}={metrics[primary]:.4f}, runtime={runtime:.1f}s", flush=True)


def configure_tensorflow(args) -> None:
    tf = import_tensorflow()
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "cuda" and not gpus:
        raise SystemExit("CUDA/GPU requested but TensorFlow cannot see a GPU.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--architecture", default="HART", choices=["HART"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-val-ratio", type=float, default=0.90)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--frame-length", type=int, default=16)
    parser.add_argument("--time-step", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--filter-attention-head", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--mlp-head-units", type=int, default=1024)
    parser.add_argument("--conv-kernels", nargs="+", type=int, default=[3, 7, 15, 31, 31, 31])
    parser.add_argument("--token-based", action="store_true")
    parser.add_argument("--channel-adapter", default="first6", choices=["first6"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 < args.train_val_ratio < 1.0:
        raise SystemExit("--train-val-ratio must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_tensorflow(args)
    print(f"Using official HART repo: {OFFICIAL_REPO} @ {OFFICIAL_COMMIT}", flush=True)
    print("TensorFlow metadata:", json.dumps(tensorflow_metadata(), ensure_ascii=False), flush=True)
    for benchmark_id in args.datasets:
        run_one(benchmark_id, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
