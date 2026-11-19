"""Run TinyHAR official code under the paper Figure 2 protocol.

The existing TinyHAR adapter runs IoTBench-HAR canonical windows. This script is
for the calibration gate: it uses the upstream ISWC22-HAR data loaders and
training loop as closely as possible, then compares macro-F1 against the values
reported in TinyHAR Figure 2.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "experiments" / "datasets"
RUNS_DIR = ROOT / "experiments" / "runs"
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "ISWC22-HAR"
OFFICIAL_REPO_URL = "https://github.com/teco-kit/ISWC22-HAR"
OFFICIAL_COMMIT = "b84b89d09f6914fe93e82cde423e294042da2741"
PAPER_ROOT = DATASETS_DIR / "tinyhar_paper_root"
PAPER_REFERENCES = {
    "dsads": {
        "official_name": "dsads",
        "benchmark_id": "dsads",
        "score": 0.8837,
        "std": 0.0115,
        "source": "TinyHAR paper Figure 2, DSADS macro-F1 88.37% +/- 1.15%",
    },
    "wisdm": {
        "official_name": "wisdm",
        "benchmark_id": "wisdm",
        "score": 0.8347,
        "std": 0.0089,
        "source": "TinyHAR paper Figure 2, WISDM macro-F1 83.47% +/- 0.89%",
    },
    "pamap2": {
        "official_name": "pamap2",
        "benchmark_id": "pamap2",
        "score": 0.7527,
        "std": 0.0087,
        "source": "TinyHAR paper Figure 2, PAMAP2 macro-F1 75.27% +/- 0.87%",
    },
    "oppo": {
        "official_name": "oppo",
        "benchmark_id": "opportunity_a",
        "score": 0.3890,
        "std": 0.0122,
        "source": "TinyHAR paper Figure 2, Opportunity macro-F1 38.90% +/- 1.22%",
    },
}
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
GAP_THRESHOLD = 0.05


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def symlink_force(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def prepare_paper_root() -> dict[str, str]:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    wisdm_src = DATASETS_DIR / "wisdm" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
    wisdm_clean_dir = PAPER_ROOT / "_clean_WISDM_Dataset"
    if wisdm_src.exists():
        wisdm_clean_dir.mkdir(parents=True, exist_ok=True)
        wisdm_clean = wisdm_clean_dir / "WISDM_ar_v1.1_raw.txt"
        kept = 0
        skipped = 0
        with wisdm_src.open("r", encoding="utf-8", errors="ignore") as fin, wisdm_clean.open("w", encoding="utf-8") as fout:
            for line in fin:
                if line.count(",") == 5:
                    fout.write(line)
                    kept += 1
                else:
                    skipped += 1
        (wisdm_clean_dir / "cleaning_note.json").write_text(
            json.dumps(
                {
                    "source": str(wisdm_src),
                    "rule": "keep rows with exactly 6 comma-separated fields; official WISDM loader parses the cleaned raw file unchanged after this.",
                    "kept_rows": kept,
                    "skipped_rows": skipped,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    dsads_root = DATASETS_DIR / "DSADS_Dataset"
    dsads_src = dsads_root / "data" if (dsads_root / "data").exists() else dsads_root
    links = {
        "WISDM_Dataset": wisdm_clean_dir if wisdm_src.exists() else DATASETS_DIR / "wisdm" / "WISDM_ar_v1.1",
        "PAMAP2_Dataset": DATASETS_DIR / "pamap2" / "PAMAP2_Dataset",
        "Opportunity_Dataset": DATASETS_DIR / "opportunity" / "OpportunityUCIDataset",
        "DSADS_Dataset": dsads_src,
    }
    status = {}
    for name, src in links.items():
        dst = PAPER_ROOT / name
        if src.exists():
            symlink_force(src.resolve(), dst)
            status[name] = str(src)
        else:
            status[name] = f"missing: {src}"
    return status


def official_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=OFFICIAL_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return OFFICIAL_COMMIT


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_config() -> dict:
    import yaml

    path = OFFICIAL_REPO / "configs" / "data.yaml"
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)


def build_args(dataset: str, seed: int, args) -> dotdict:
    config = load_data_config()[dataset]
    run_args = dotdict()
    run_args.to_save_path = str(RUNS_DIR / args.run_tag / "official_logs" / dataset / f"seed_{seed}")
    run_args.freq_save_path = str(RUNS_DIR / args.run_tag / "official_freq")
    run_args.window_save_path = str(RUNS_DIR / args.run_tag / "official_windows")
    run_args.root_path = str(PAPER_ROOT / config["filename"])
    run_args.drop_transition = False
    run_args.datanorm_type = "standardization"
    run_args.batch_size = args.batch_size
    run_args.shuffle = True
    run_args.drop_last = False
    run_args.train_vali_quote = 0.90
    run_args.train_epochs = args.epochs
    run_args.learning_rate = args.lr
    run_args.learning_rate_patience = 7
    run_args.learning_rate_factor = 0.1
    run_args.early_stop_patience = args.early_stop_patience
    run_args.use_gpu = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    run_args.gpu = args.gpu
    run_args.use_multi_gpu = False
    run_args.optimizer = "Adam"
    run_args.criterion = "CrossEntropy"
    run_args.seed = seed
    run_args.data_name = dataset
    run_args.wavelet_filtering = False
    run_args.wavelet_filtering_regularization = False
    run_args.wavelet_filtering_finetuning = False
    run_args.wavelet_filtering_finetuning_percent = 0.5
    run_args.wavelet_filtering_learnable = False
    run_args.wavelet_filtering_layernorm = False
    run_args.regulatization_tradeoff = 0
    run_args.number_wavelet_filtering = 12
    run_args.difference = False
    run_args.filtering = False
    run_args.magnitude = False
    run_args.weighted_sampler = False
    run_args.pos_select = None
    run_args.sensor_select = None
    run_args.representation_type = "time"
    run_args.exp_mode = "LOCV" if dataset != "skodar" else "SOCV"
    run_args.sampling_freq = config["sampling_freq"]
    run_args.num_classes = config["num_classes"]
    run_args.windowsize = int(config["window_seconds"] * config["sampling_freq"])
    run_args.input_length = run_args.windowsize
    run_args.c_in = config["num_channels"]
    run_args.f_in = 1
    run_args.model_type = "tinyhar"
    run_args.cross_channel_interaction_type = "attn"
    run_args.cross_channel_aggregation_type = "FC"
    run_args.temporal_info_interaction_type = "lstm"
    run_args.temporal_info_aggregation_type = "tnaive"
    run_args.filter_scaling_factor = 1
    run_args.mixup = False
    run_args.alpha = 0.2
    run_args.output_attention = False
    run_args.sample_wise = False
    return run_args


def import_experiment():
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    if str(OFFICIAL_REPO / "notebooks" / "model") not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO / "notebooks" / "model"))
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        plot_stub = types.SimpleNamespace(
            figure=lambda *args, **kwargs: None,
            savefig=lambda *args, **kwargs: None,
            close=lambda *args, **kwargs: None,
            plot=lambda *args, **kwargs: None,
            imshow=lambda *args, **kwargs: None,
            show=lambda *args, **kwargs: None,
        )
        mpl_stub = types.SimpleNamespace(pyplot=plot_stub, pylab=plot_stub)
        sys.modules["matplotlib"] = mpl_stub
        sys.modules["matplotlib.pyplot"] = plot_stub
        sys.modules["matplotlib.pylab"] = plot_stub
    try:
        import seaborn  # noqa: F401
    except Exception:
        sys.modules["seaborn"] = types.SimpleNamespace(heatmap=lambda *args, **kwargs: None)
    from experiment import Exp  # type: ignore

    return Exp


def parse_score_file(path: Path) -> list[dict]:
    rows = []
    pattern = re.compile(r"Test weighted F1:\s*([0-9.]+)\s+Test macro F1\s*([0-9.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"f1_weighted": float(match.group(1)), "f1_macro": float(match.group(2))})
    return rows


def find_score_file(to_save_path: Path) -> Path:
    score_files = sorted(to_save_path.glob("logs/tinyhar_data_*/score.txt"))
    if not score_files:
        raise SystemExit(f"Missing official score.txt under {to_save_path}")
    if len(score_files) > 1:
        score_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return score_files[0]


def run_seed(dataset: str, seed: int, args) -> dict:
    Exp = import_experiment()
    run_args = build_args(dataset, seed, args)
    to_save_path = Path(run_args.to_save_path)
    if to_save_path.exists() and args.clean:
        shutil.rmtree(to_save_path)
    to_save_path.mkdir(parents=True, exist_ok=True)
    Path(run_args.freq_save_path).mkdir(parents=True, exist_ok=True)
    Path(run_args.window_save_path).mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    start = time.time()
    old_cwd = Path.cwd()
    os.chdir(OFFICIAL_REPO / "notebooks" / "model")
    try:
        exp = Exp(run_args)
        exp.train()
    finally:
        os.chdir(old_cwd)
    score_file = find_score_file(to_save_path)
    folds = parse_score_file(score_file)
    if not folds:
        raise SystemExit(f"No TinyHAR score rows parsed from {score_file}")
    macro = np.asarray([row["f1_macro"] for row in folds], dtype=np.float64)
    weighted = np.asarray([row["f1_weighted"] for row in folds], dtype=np.float64)
    return {
        "seed": seed,
        "folds": len(folds),
        "score_file": str(score_file.relative_to(ROOT)),
        "runtime_seconds": round(time.time() - start, 3),
        "metrics": {
            "f1_macro": float(macro.mean()),
            "f1_macro_fold_std": float(macro.std(ddof=0)),
            "f1_weighted": float(weighted.mean()),
            "f1_weighted_fold_std": float(weighted.std(ddof=0)),
        },
        "fold_metrics": folds,
    }


def blocker_entry(dataset: str, message: str, link_status: dict) -> dict:
    ref = PAPER_REFERENCES[dataset]
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": None,
        "primary_metric": "f1_macro",
        "metrics": {},
        "source": "official_code_paper_protocol",
        "protocol_status": "blocked",
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": official_commit(),
        "paper_reference_score": ref["score"],
        "paper_reference_std": ref["std"],
        "paper_reference_metric": "f1_macro",
        "paper_reference_source": ref["source"],
        "paper_reference_gap_status": "blocked_missing_dataset",
        "calibration_status": "blocked",
        "calibration_note": message,
        "paper_protocol_dataset_name": ref["official_name"],
        "paper_data_links": link_status,
    }
    return {
        "id": "tinyhar",
        "method_name": "TinyHAR",
        "year": 2022,
        "method_family": "Lightweight CNN/RNN for edge HAR",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_calibration_blocked",
        "summary": "TinyHAR official paper-protocol calibration could not run because a required dataset is missing.",
        "implementation": "Official ISWC22-HAR TinyHAR training loop and data loaders.",
        "har_scope": "TinyHAR paper Figure 2 LOSO macro-F1 calibration.",
        "datasets": [],
        "results": [result],
        "sources": [OFFICIAL_REPO_URL, "https://publikationen.bibliothek.kit.edu/1000150216/149250239"],
    }


def make_entry(dataset: str, seed_rows: list[dict], args, link_status: dict) -> dict:
    ref = PAPER_REFERENCES[dataset]
    seed_macro = np.asarray([row["metrics"]["f1_macro"] for row in seed_rows], dtype=np.float64)
    seed_weighted = np.asarray([row["metrics"]["f1_weighted"] for row in seed_rows], dtype=np.float64)
    score = float(seed_macro.mean())
    gap = score - ref["score"]
    passed = abs(gap) <= args.gap_threshold
    result = {
        "benchmark_id": ref["benchmark_id"],
        "primary_score": score,
        "primary_metric": "f1_macro",
        "metrics": {
            "f1_macro": score,
            "f1_macro_std": float(seed_macro.std(ddof=0)),
            "f1_weighted": float(seed_weighted.mean()),
            "f1_weighted_std": float(seed_weighted.std(ddof=0)),
        },
        "source": "official_code_paper_protocol",
        "protocol_status": "paper_protocol",
        "official_code_used": True,
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": official_commit(),
        "official_model_class": "models.TinyHAR.TinyHAR_Model",
        "official_training_loop": "experiment.Exp.train",
        "official_configuration_source": "configs/data.yaml + configs/model.yaml + notebooks/model/Train model.ipynb",
        "paper_protocol_dataset_name": ref["official_name"],
        "paper_protocol": {
            "exp_mode": "LOCV",
            "seeds": [row["seed"] for row in seed_rows],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_vali_quote": 0.90,
            "early_stop_patience": args.early_stop_patience,
            "normalization": "official datanorm_type=standardization, fit on train/validation subjects per LOCV fold",
            "model": "TinyHAR attn + FC cross-channel fusion + LSTM + tnaive temporal aggregation, filter_num=20",
        },
        "paper_reference_score": ref["score"],
        "paper_reference_std": ref["std"],
        "paper_reference_metric": "f1_macro",
        "paper_reference_source": ref["source"],
        "paper_reference_gap_signed": gap,
        "paper_reference_gap_abs": abs(gap),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": "within_expected_range" if passed else "large_gap_check_protocol",
        "calibration_status": "passed" if passed else "failed",
        "seed_results": seed_rows,
        "paper_data_links": link_status,
    }
    return {
        "id": "tinyhar",
        "method_name": "TinyHAR",
        "year": 2022,
        "method_family": "Lightweight CNN/RNN for edge HAR",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_calibration",
        "summary": "TinyHAR official-code paper-protocol calibration against Figure 2 macro-F1.",
        "implementation": "Runs the upstream ISWC22-HAR Exp/TinyHAR_Model path with the paper notebook hyperparameters and official data loaders.",
        "har_scope": "TinyHAR paper Figure 2 LOSO macro-F1 calibration before Standard promotion.",
        "datasets": [
            {
                "benchmark_id": ref["benchmark_id"],
                "dataset": ref["official_name"],
                "reported_effect": f"Paper macro-F1 {ref['score']:.4f} +/- {ref['std']:.4f}; official rerun {score:.4f}.",
                "protocol_note": "TinyHAR paper protocol: official ISWC22-HAR loader, LOCV, five seeds, macro-F1.",
            }
        ],
        "results": [result],
        "sources": [OFFICIAL_REPO_URL, "https://publikationen.bibliothek.kit.edu/1000150216/149250239"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wisdm", "pamap2", "oppo", "dsads"], choices=sorted(PAPER_REFERENCES))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--run-tag", default="tinyhar_paper_official_calibration")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)

    link_status = prepare_paper_root()
    entries = []
    for dataset in args.datasets:
        official_name = PAPER_REFERENCES[dataset]["official_name"]
        expected_path = PAPER_ROOT / load_data_config()[official_name]["filename"]
        if not expected_path.exists():
            message = f"TinyHAR paper dataset path is missing: {expected_path}"
            print(message, flush=True)
            entries.append(blocker_entry(dataset, message, link_status))
            continue
        seed_rows = []
        for seed in args.seeds:
            print(f"TinyHAR paper calibration dataset={dataset} seed={seed}", flush=True)
            seed_rows.append(run_seed(official_name, seed, args))
        entries.append(make_entry(dataset, seed_rows, args, link_status))

    out_path = RUNS_DIR / args.run_tag / "results.json"
    write_json(out_path, entries)
    print(f"Wrote {out_path}", flush=True)
    if args.merge:
        subprocess.check_call([sys.executable, "scripts/merge_experiment_results.py", str(out_path)], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
