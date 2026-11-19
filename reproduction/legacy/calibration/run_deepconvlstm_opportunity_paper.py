"""Run DeepConvLSTM on OPPORTUNITY with the original paper protocol.

The official STRCWearlab repository publishes preprocessing code and a gesture
checkpoint, but not a locomotion checkpoint. This runner keeps the paper/official
OPPORTUNITY protocol fixed and trains the DeepConvLSTM architecture for the
requested OPPORTUNITY task.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


if not hasattr(np, "product"):
    np.product = np.prod

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "DeepConvLSTM_official"
OPPORTUNITY_ROOT = ROOT / "experiments" / "datasets" / "opportunity"
RUNS_DIR = ROOT / "experiments" / "runs"

NB_SENSOR_CHANNELS = 113
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12
FINAL_SEQUENCE_LENGTH = 8
BATCH_SIZE = 100
NUM_FILTERS = 64
FILTER_SIZE = 5
NUM_UNITS_LSTM = 128
TRAIN_ROWS = 557963

PAPER_REFERENCES = {
    "locomotion": {
        "classes": 5,
        "table": "DeepConvLSTM paper Table 4, OPPORTUNITY modes of locomotion",
        "f1_weighted": 0.930,
        "f1_weighted_no_null": 0.895,
    },
    "gestures": {
        "classes": 18,
        "table": "DeepConvLSTM paper Table 4, OPPORTUNITY gesture recognition",
        "f1_weighted": 0.915,
        "f1_weighted_no_null": 0.866,
    },
}

BENCHMARK_IDS = {
    "gestures": "opportunity_a",
    "locomotion": "opportunity_b",
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def load_py2_module(path: Path, module_name: str):
    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    source = path.read_text(encoding="utf-8")
    if not source.endswith("\n"):
        source += "\n"
    converted = str(tool.refactor_string(source, str(path)))
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(converted, str(path), "exec"), module.__dict__)
    return module


def load_official_modules():
    sys.path.insert(0, str(OFFICIAL_REPO))
    preprocess = load_py2_module(OFFICIAL_REPO / "preprocess_data.py", "official_deepconvlstm_preprocess_train")
    sliding = load_py2_module(OFFICIAL_REPO / "sliding_window.py", "official_deepconvlstm_sliding_train")
    return preprocess, sliding.sliding_window


def load_official_split(dataset_root: Path, label: str):
    preprocess, _sliding_window = load_official_modules()
    data_x = np.empty((0, NB_SENSOR_CHANNELS), dtype=np.float32)
    data_y = np.empty((0,), dtype=np.uint8)
    for relative_name in preprocess.OPPORTUNITY_DATA_FILES:
        path = dataset_root / relative_name
        if not path.is_file():
            raise SystemExit(f"Missing OPPORTUNITY file required by official DeepConvLSTM: {path}")
        print(f"official preprocess {relative_name}", flush=True)
        raw = np.loadtxt(str(path))
        x_part, y_part = preprocess.process_dataset_file(raw, label)
        data_x = np.vstack((data_x, x_part.astype(np.float32)))
        data_y = np.concatenate((data_y, y_part.astype(np.uint8)))
    if len(data_x) <= TRAIN_ROWS:
        raise SystemExit(f"Official OPPORTUNITY split has no test rows; got {len(data_x)}")
    return data_x[:TRAIN_ROWS], data_y[:TRAIN_ROWS], data_x[TRAIN_ROWS:], data_y[TRAIN_ROWS:]


def official_sliding_windows(data_x: np.ndarray, data_y: np.ndarray):
    _preprocess, sliding_window = load_official_modules()
    x = sliding_window(data_x, (SLIDING_WINDOW_LENGTH, data_x.shape[1]), (SLIDING_WINDOW_STEP, 1))
    y = np.asarray([[i[-1]] for i in sliding_window(data_y, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)])
    x = x.astype(np.float32).reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    y = y.reshape(len(y)).astype(np.uint8)
    return x, y


class DeepConvLSTMPaper(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, NUM_FILTERS, (FILTER_SIZE, 1)),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1)),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1)),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1)),
            nn.ReLU(),
        )
        self.drop_in = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            NUM_FILTERS * NB_SENSOR_CHANNELS,
            NUM_UNITS_LSTM,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )
        self.drop_out = nn.Dropout(0.5)
        self.fc = nn.Linear(NUM_UNITS_LSTM, classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Lasagne LSTMLayer flattens trailing dimensions after time.
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        out, _ = self.lstm(self.drop_in(x))
        return self.fc(self.drop_out(out[:, -1]))


def iter_full_batches(x: np.ndarray, y: np.ndarray, batch_size: int):
    usable = len(x) - (len(x) % batch_size)
    for start in range(0, usable, batch_size):
        stop = start + batch_size
        yield x[start:stop], y[start:stop]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=sorted(set(y_true.tolist())))),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true.tolist())))),
    }
    mask = y_true != 0
    if mask.any():
        labels = sorted(set(y_true[mask].tolist()))
        out["accuracy_no_null"] = float(accuracy_score(y_true[mask], y_pred[mask]))
        out["f1_weighted_no_null"] = float(f1_score(y_true[mask], y_pred[mask], average="weighted", labels=labels))
        out["f1_macro_no_null"] = float(f1_score(y_true[mask], y_pred[mask], average="macro", labels=labels))
    return out


def train_eval(x_train, y_train, x_test, y_test, classes: int, args, device: torch.device):
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    model = DeepConvLSTMPaper(classes).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9)
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            current = evaluate(model, x_test, y_test, device)
            best = current if best is None or current["f1_weighted"] > best["f1_weighted"] else best
            print(
                "epoch={} loss={:.4f} f1_weighted={:.4f} f1_no_null={:.4f} accuracy={:.4f}".format(
                    epoch,
                    float(np.mean(losses)),
                    current["f1_weighted"],
                    current.get("f1_weighted_no_null", 0.0),
                    current["accuracy"],
                ),
                flush=True,
            )
        else:
            print(f"epoch={epoch} loss={float(np.mean(losses)):.4f}", flush=True)
    final = evaluate(model, x_test, y_test, device)
    return final, best or final, time.time() - start


def evaluate(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, device: torch.device) -> dict:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in iter_full_batches(x_test, y_test, BATCH_SIZE):
            logits = model(torch.from_numpy(xb).to(device, non_blocking=True))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb)
    y_pred = np.concatenate(preds).astype(np.uint8)
    y_true = np.concatenate(trues).astype(np.uint8)
    out = metrics(y_true, y_pred)
    out["evaluated_windows"] = int(len(y_true))
    out["dropped_incomplete_batch_windows"] = int(len(x_test) - len(y_true))
    return out


def write_results(task: str, final_metrics: dict, best_metrics: dict, runtime: float, args, device: torch.device) -> None:
    ref = PAPER_REFERENCES[task]
    benchmark_id = BENCHMARK_IDS[task]
    gaps = {
        "f1_weighted": final_metrics["f1_weighted"] - ref["f1_weighted"],
        "f1_weighted_no_null": final_metrics.get("f1_weighted_no_null", 0.0) - ref["f1_weighted_no_null"],
    }
    result = {
        "benchmark_id": benchmark_id,
        "primary_score": final_metrics["f1_weighted"],
        "metrics": final_metrics,
        "best_checkpoint_metrics": best_metrics,
        "source": "paper_protocol_local_training",
        "protocol_status": "paper_protocol",
        "official_code_used_for_preprocessing": True,
        "official_architecture_reimplemented": "PyTorch Conv2D valid 5x1 + 2-layer LSTM equivalent to the Lasagne notebook; no released official checkpoint for this task.",
        "paper_reference_metric": "f1_weighted",
        "paper_reference_score": ref["f1_weighted"],
        "paper_reference_metrics": {
            "f1_weighted": ref["f1_weighted"],
            "f1_weighted_no_null": ref["f1_weighted_no_null"],
        },
        "paper_reference_source": ref["table"],
        "paper_reference_metric_gaps": gaps,
        "paper_reference_max_gap_abs": float(max(abs(value) for value in gaps.values())),
        "paper_reference_gap_threshold": args.gap_threshold,
        "paper_reference_gap_status": (
            "within_expected_range"
            if max(abs(value) for value in gaps.values()) <= args.gap_threshold
            else "large_gap_check_protocol"
        ),
        "runtime_seconds": round(runtime, 3),
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "window_size_samples": SLIDING_WINDOW_LENGTH,
        "stride_samples": SLIDING_WINDOW_STEP,
        "batch_size": BATCH_SIZE,
        "classes": ref["classes"],
    }
    if device.type == "cuda":
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(device)
    entry = {
        "id": "deepconvlstm",
        "method_name": "DeepConvLSTM",
        "year": 2016,
        "method_family": "CNN + LSTM",
        "submission_type": "platform",
        "coverage_status": "paper_protocol_local_training",
        "summary": f"DeepConvLSTM OPPORTUNITY {task} paper-protocol training run.",
        "implementation": "Paper architecture and official preprocessing/split/window protocol; trained locally because no official checkpoint is released for this task.",
        "har_scope": "OPPORTUNITY challenge wearable HAR paper protocol.",
        "datasets": [
            {
                "benchmark_id": benchmark_id,
                "dataset": "OPPORTUNITY Activity Recognition",
                "activity_task": task,
                "subjects": "Official OPPORTUNITY challenge file order and 557963-row train/test split.",
                "device_placement": "113 OPPORTUNITY challenge sensor channels.",
                "reported_effect": "Paper-protocol local training f1_weighted {:.4f} vs paper {:.4f}.".format(
                    final_metrics["f1_weighted"], ref["f1_weighted"]
                ),
                "protocol_note": "500 ms windows, 250 ms stride, last-sample label, batch size 100, RMSProp lr=1e-3 alpha=0.9.",
            }
        ],
        "results": [result],
        "sources": [
            "https://github.com/STRCWearlab/DeepConvLSTM",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/",
        ],
    }
    out_dir = RUNS_DIR / f"opportunity_{task}_deepconvlstm_paper_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'results.json'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(PAPER_REFERENCES), default="locomotion")
    parser.add_argument("--dataset-root", default=str(OPPORTUNITY_ROOT))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    seed_all(args.seed)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device: {device}", flush=True)

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_official_split(Path(args.dataset_root), args.task)
    x_train, y_train = official_sliding_windows(x_train_raw, y_train_raw)
    x_test, y_test = official_sliding_windows(x_test_raw, y_test_raw)
    print(f"official windows train={x_train.shape} test={x_test.shape}", flush=True)

    final, best, runtime = train_eval(x_train, y_train, x_test, y_test, PAPER_REFERENCES[args.task]["classes"], args, device)
    write_results(args.task, final, best, runtime, args, device)
    print(
        "DeepConvLSTM paper protocol {}: f1_weighted={:.4f}, f1_no_null={:.4f}, accuracy={:.4f}, runtime={:.1f}s".format(
            args.task,
            final["f1_weighted"],
            final.get("f1_weighted_no_null", 0.0),
            final["accuracy"],
            runtime,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
