"""Run LIMU-BERT official pipeline on additional IoTBench-HAR datasets.

This is the post-calibration extension path. UCI-HAR and HHAR are handled by
run_limu_bert_paper_official.py because they have directly comparable paper
Table 2 references. For the remaining Standard datasets, this runner keeps the
same official LIMU-BERT stages:

1. convert canonical benchmark windows into the official 20_120 numpy layout,
2. run the upstream pretrain.py logic on unlabeled windows,
3. generate embeddings with the upstream embedding.py logic,
4. train LIMU-GRU with the upstream classifier.py logic.

The output is marked as an official-code extension rather than a paper table
reproduction because these datasets were not reported in the LIMU-BERT paper.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "LIMU-BERT-Public"
OFFICIAL_REPO_URL = "https://github.com/dapowan/LIMU-BERT-Public"
OFFICIAL_COMMIT_FALLBACK = "decffee7ecb4e2e5d7244b1a759cb80b752dc6c2"
GAP_THRESHOLD = 0.05

EXTENSION_DATASETS = ["wisdm", "pamap2", "usc_had", "opportunity_a", "opportunity_b", "realworld_har"]
PRIMARY_METRIC = {
    "wisdm": "f1_macro",
    "pamap2": "f1_weighted",
    "usc_had": "f1_macro",
    "opportunity_a": "f1_weighted",
    "opportunity_b": "f1_weighted",
    "realworld_har": "f1_macro",
}


def import_project():
    if str(ROOT / "experiments") not in sys.path:
        sys.path.insert(0, str(ROOT / "experiments"))
    from har_utils import load_windows, write_json

    return load_windows, write_json


def import_official():
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    pretrain = importlib.import_module("pretrain")
    embedding_mod = importlib.import_module("embedding")
    classifier_mod = importlib.import_module("classifier")
    config_mod = importlib.import_module("config")
    return pretrain, embedding_mod, classifier_mod, config_mod


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


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=cwd or ROOT)
    if code != 0:
        raise SystemExit(f"Command failed with exit {code}: {' '.join(cmd)}")


def wait_for_gpu(max_util: int, poll_seconds: int) -> None:
    while True:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            util = max(int(float(x.strip())) for x in out.splitlines() if x.strip())
        except Exception:
            return
        if util <= max_util:
            return
        print(f"GPU busy ({util}%); waiting {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)


def resample_length(x: np.ndarray, target_len: int = 120) -> np.ndarray:
    if x.shape[1] == target_len:
        return x.astype(np.float32)
    old_t = np.linspace(0.0, 1.0, x.shape[1], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((x.shape[0], target_len, x.shape[2]), dtype=np.float32)
    for c in range(x.shape[2]):
        out[:, :, c] = np.apply_along_axis(lambda row: np.interp(new_t, old_t, row), 1, x[:, :, c])
    return out


def adapt_channels(x: np.ndarray) -> tuple[np.ndarray, str]:
    channels = x.shape[2]
    if channels == 6:
        return x.astype(np.float32), "native_6ch"
    if channels > 6:
        return x[:, :, :6].astype(np.float32), f"first6_of_{channels}ch"
    pad = np.zeros((x.shape[0], x.shape[1], 6 - channels), dtype=np.float32)
    return np.concatenate([x, pad], axis=2).astype(np.float32), f"zero_padded_{channels}ch_to_6ch"


def make_labels(y: np.ndarray, subjects: np.ndarray, seq_len: int) -> tuple[np.ndarray, dict[int, int]]:
    subject_values = sorted(int(v) for v in np.unique(subjects))
    subject_map = {value: idx for idx, value in enumerate(subject_values)}
    mapped_subjects = np.asarray([subject_map[int(v)] for v in subjects], dtype=np.int64)
    labels = np.zeros((len(y), seq_len, 2), dtype=np.float32)
    labels[:, :, 0] = y.astype(np.float32)[:, None]
    labels[:, :, 1] = mapped_subjects.astype(np.float32)[:, None]
    return labels, subject_map


def update_data_config(dataset_name: str, data: np.ndarray, labels: np.ndarray, class_count: int, subject_count: int) -> None:
    path = OFFICIAL_REPO / "dataset" / "data_config.json"
    cfg = json.loads(path.read_text())
    cfg[f"{dataset_name}_20_120"] = {
        "sr": 20,
        "seq_len": 120,
        "dimension": int(data.shape[2]),
        "activity_label_index": 0,
        "activity_label_size": int(class_count),
        "activity_label": [f"class_{i}" for i in range(class_count)],
        "user_label_index": 1,
        "user_label_size": int(subject_count),
        "size": int(data.shape[0]),
    }
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=4) + "\n")


def prepare_official_dataset(benchmark_id: str, *, force: bool = False) -> dict:
    load_windows, _ = import_project()
    official_name = f"iotbench_{benchmark_id}"
    out_dir = OFFICIAL_REPO / "dataset" / official_name
    data_path = out_dir / "data_20_120.npy"
    label_path = out_dir / "label_20_120.npy"
    meta_path = out_dir / "iotbench_meta.json"
    if data_path.is_file() and label_path.is_file() and meta_path.is_file() and not force:
        return json.loads(meta_path.read_text())

    x, y, subjects, meta = load_windows(benchmark_id)
    x = resample_length(x, 120)
    x, channel_adapter = adapt_channels(x)
    labels, subject_map = make_labels(y, subjects, 120)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_path, x.astype(np.float32))
    np.save(label_path, labels.astype(np.float32))
    update_data_config(official_name, x, labels, len(np.unique(y)), len(subject_map))
    prepared = {
        "benchmark_id": benchmark_id,
        "official_dataset_name": official_name,
        "data_path": str(data_path),
        "label_path": str(label_path),
        "samples": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "dimension": int(x.shape[2]),
        "classes": int(len(np.unique(y))),
        "subjects": int(len(subject_map)),
        "channel_adapter": channel_adapter,
        "source_window_shape": list(load_windows(benchmark_id)[0].shape),
        "conversion_note": "Canonical windows converted to official LIMU-BERT 20_120 numpy layout; labels repeat activity and subject ids across timesteps.",
    }
    meta_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n")
    return prepared


def make_train_config(src_name: str, out_name: str, epochs: int) -> Path:
    base = json.loads((OFFICIAL_REPO / "config" / src_name).read_text())
    base["n_epochs"] = int(epochs)
    out = OFFICIAL_REPO / "config" / out_name
    out.write_text(json.dumps(base, indent=2) + "\n")
    return out


def make_args(target: str, dataset_name: str, save_name: str, train_cfg: Path, *, model_file: str | None = None, gpu: int = 0):
    _, _, _, config_mod = import_official()
    model_cfg = config_mod.load_model_config(target, "base" if "pretrain" in target else "gru", "v1" if "pretrain" in target else "v2")
    dataset_cfg = config_mod.load_dataset_stats(dataset_name, "20_120")
    if model_cfg is None or dataset_cfg is None:
        raise SystemExit(f"Missing official config for target={target} dataset={dataset_name}")
    save_dir = OFFICIAL_REPO / "saved" / f"{target}_{dataset_name}_20_120"
    save_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        model_version="v1" if "pretrain" in target else "v2",
        dataset=dataset_name,
        dataset_version="20_120",
        gpu=str(gpu),
        model_file=model_file,
        train_cfg=str(train_cfg),
        mask_cfg=str(OFFICIAL_REPO / "config" / "mask.json"),
        label_index=0,
        save_model=save_name,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        data_path=str(OFFICIAL_REPO / "dataset" / dataset_name / "data_20_120.npy"),
        label_path=str(OFFICIAL_REPO / "dataset" / dataset_name / "label_20_120.npy"),
        save_path=str(save_dir / save_name),
        pretrain_model=(str(save_dir / model_file) if model_file else None),
    )


def parse_best(lines: list[str]) -> dict:
    import re

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


class Tee:
    def __init__(self):
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        for line in text.splitlines():
            self.lines.append(line)

    def flush(self) -> None:
        sys.__stdout__.flush()


def run_dataset(benchmark_id: str, args) -> None:
    if args.wait_for_gpu:
        wait_for_gpu(args.max_gpu_util, args.poll_seconds)
    pretrain, embedding_mod, classifier_mod, _ = import_official()
    load_windows, write_json = import_project()
    prepared = prepare_official_dataset(benchmark_id, force=args.force_prepare)
    official_name = prepared["official_dataset_name"]

    pretrain_cfg = make_train_config("pretrain.json", f"pretrain_iotbench_{args.pretrain_epochs}.json", args.pretrain_epochs)
    classifier_cfg = make_train_config("train.json", f"train_iotbench_{args.classifier_epochs}.json", args.classifier_epochs)
    pretrain_name = f"{official_name}_pretrain_{args.pretrain_epochs}"
    classifier_name = f"{official_name}_gru_{args.classifier_epochs}"
    start = time.time()

    pretrain_args = make_args("pretrain_base", official_name, pretrain_name, pretrain_cfg, gpu=args.gpu)
    pretrain_ckpt = Path(pretrain_args.save_path + ".pt")
    if args.force_pretrain or not pretrain_ckpt.is_file():
        pretrain.main(pretrain_args, training_rate=0.8)

    embed_path = OFFICIAL_REPO / "embed" / f"embed_{pretrain_name}_{official_name}_20_120.npy"
    embed_args = make_args("pretrain_base", official_name, pretrain_name, pretrain_cfg, model_file=pretrain_name, gpu=args.gpu)
    embed_args.pretrain_model = str(pretrain_ckpt.with_suffix(""))
    if args.force_embedding or not embed_path.is_file():
        (OFFICIAL_REPO / "embed").mkdir(exist_ok=True)
        embedding_mod.generate_embedding_or_output(args=embed_args, output_embed=True, save=True)

    classifier_args = make_args("classifier_base_gru", official_name, classifier_name, classifier_cfg, model_file=pretrain_name, gpu=args.gpu)
    tee = Tee()
    old_stdout = sys.stdout
    try:
        sys.stdout = tee
        embedding, labels = embedding_mod.load_embedding_label(classifier_args.model_file, classifier_args.dataset, classifier_args.dataset_version)
        label_test, label_estimate_test = classifier_mod.classify_embeddings(
            classifier_args,
            embedding,
            labels,
            classifier_args.label_index,
            training_rate=0.8,
            label_rate=0.01,
            balance=True,
            method="gru",
        )
        classifier_mod.stat_results(label_test, label_estimate_test)
    finally:
        sys.stdout = old_stdout

    metrics = parse_best(tee.lines)
    primary = PRIMARY_METRIC[benchmark_id]
    runtime = round(time.time() - start, 3)
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": float(metrics[primary]),
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
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": official_commit(),
        "official_pipeline": "pretrain.py -> embedding.py -> classifier.py",
        "official_dataset": f"dataset/{official_name}/data_20_120.npy and label_20_120.npy",
        "official_model_version_pretrain": "v1",
        "official_model_version_classifier": "v2",
        "official_classifier": "classifier.py v2/base_gru",
        "official_classifier_config": "config/classifier.json: gru_v2",
        "official_pretrain_config": f"config/pretrain_iotbench_{args.pretrain_epochs}.json derived from config/pretrain.json",
        "official_train_config": f"config/train_iotbench_{args.classifier_epochs}.json derived from config/train.json",
        "official_training_rate": 0.8,
        "official_validation_rate": 0.1,
        "official_label_rate": 0.01,
        "official_balance": True,
        "official_train_seed": 3431,
        "pretrain_epochs": args.pretrain_epochs,
        "epochs": args.classifier_epochs,
        "runtime_seconds": runtime,
        "calibration_status": "extension_after_uci_hhar_calibration_passed",
        "paper_reference_gap_status": "unavailable_for_same_benchmark_metric",
        "channel_adapter": prepared["channel_adapter"],
        "official_format_preparation": prepared,
        "official_stdout_tail": tee.lines[-120:],
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
        "sources": [OFFICIAL_REPO_URL, "https://dl.acm.org/doi/10.1145/3485730.3485937"],
    }
    out_dir = RUNS_DIR / f"{benchmark_id}_limu_bert_official_extension"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    write_json(out_dir / "results.json", [entry])
    print(f"{benchmark_id} LIMU-BERT official extension: {primary}={metrics[primary]:.4f}", flush=True)
    if args.merge:
        run_cmd([sys.executable, "scripts/merge_experiment_results.py", str(out_dir / "results.json")])
        run_cmd([sys.executable, "experiments/audit_reproduction_gaps.py"])


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wisdm"], choices=EXTENSION_DATASETS)
    parser.add_argument("--pretrain-epochs", type=int, default=3200)
    parser.add_argument("--classifier-epochs", type=int, default=700)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--max-gpu-util", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--force-embedding", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    old_cwd = Path.cwd()
    os.chdir(OFFICIAL_REPO)
    try:
        for dataset in args.datasets:
            run_dataset(dataset, args)
    finally:
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
