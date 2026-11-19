"""Evaluate the original Attention Model author checkpoint.

The paper archive contains source plus saved state_dict files named with their
reported scores. This runner keeps the author's OPPORTUNITY preprocessing,
model shape, and sample-wise test filling behavior, while replacing hard-coded
paths with IoTBench-HAR local paths.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.autograd import Variable


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
AUTHOR_REPO = ROOT / "experiments" / "repos" / "AttentionModels" / "AttentionModels"
OPPORTUNITY_ROOT = ROOT / "experiments" / "datasets" / "opportunity"

NB_SENSOR_CHANNELS = 77
NUM_CLASSES = 18
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12
TEST_SLIDING_WINDOW_STEP = 1
BATCH_SIZE = 100
NUM_FILTERS = 64
FILTER_SIZE = 5
FILTER_STRIDE = 1
NUM_UNITS_LSTM = 128
NUM_LAYERS_LSTM = 2
PAPER_SCORE = 0.707
GAP_THRESHOLD = 0.05


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AuthorAttentionRCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        is_bidirectional: bool = False,
        dropout: float = 0.79,
        attention_dropout: float = 0.74,
    ) -> None:
        super().__init__()
        self.is_bidirectional = is_bidirectional
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        hidden_dim = hidden_size * self.num_directions
        self.dropout_val = dropout
        self.attention_dropout_val = attention_dropout
        self.conv2DLayer1 = nn.Conv2d(1, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE, 1))
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE, 1))
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE, 1))
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE, 1))
        self.relu4 = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=is_bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense_layer = nn.Linear(hidden_dim, NUM_CLASSES)
        self.num_layers = num_layers
        self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.attentionLayer2 = nn.Linear(hidden_dim, 1)
        self.softmax_attention = torch.nn.Softmax(dim=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        convout1 = self.relu1(self.conv2DLayer1(input))
        convout2 = self.relu2(self.conv2DLayer2(convout1))
        convout3 = self.relu3(self.conv2DLayer3(convout2))
        convout4 = self.relu4(self.conv2DLayer4(convout3))
        lstm_input = convout4.permute(2, 0, 1, 3).contiguous()
        lstm_input = lstm_input.view(lstm_input.shape[0], lstm_input.shape[1], -1)
        lstm_input = self.dropout(lstm_input)
        output, _hidden = self.lstm(lstm_input, self.init_hidden(input.device))
        past_context = output[:-1]
        current = output[-1]
        attention_layer1_output = self.attentionLayer1(past_context)
        attention_layer1_output = attention_layer1_output + current
        attention_layer1_output = self.tanh1(attention_layer1_output)
        attention_layer1_output = self.attention_dropout(attention_layer1_output)
        attention_layer2_output = self.attentionLayer2(attention_layer1_output).squeeze(2)
        attn_weights = self.softmax_attention(attention_layer2_output).unsqueeze(2)
        new_context_vector = torch.sum(attn_weights * past_context, 0)
        new_context_vector = new_context_vector + current
        return self.dense_layer(new_context_vector)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = torch.nn.Softmax(dim=1)(self.forward(inputs))
        return torch.max(probs, 1)[1]

    def init_hidden(self, device: torch.device):
        h0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, BATCH_SIZE, self.hidden_size), 0.08)).to(device)
        c0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, BATCH_SIZE, self.hidden_size), 0.08)).to(device)
        return h0, c0


def iter_full_batches(inputs: np.ndarray, targets: np.ndarray):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - BATCH_SIZE + 1, BATCH_SIZE):
        excerpt = slice(start_idx, start_idx + BATCH_SIZE)
        yield inputs[excerpt], targets[excerpt]


def load_author_modules() -> None:
    if not AUTHOR_REPO.is_dir():
        raise SystemExit(f"Missing extracted AttentionModels archive: {AUTHOR_REPO}")
    sys.path.insert(0, str(AUTHOR_REPO))
    if not hasattr(np, "product"):
        np.product = np.prod


def ensure_processed(processed_path: Path, raw_root: Path, force: bool) -> None:
    if processed_path.is_file() and not force:
        print(f"Using existing author processed data: {processed_path}", flush=True)
        return
    load_author_modules()
    from preprocess_data import generate_data

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating author OPPORTUNITY data: {processed_path}", flush=True)
    generate_data(str(raw_root), str(processed_path), "gestures")


def load_dataset(processed_path: Path):
    with processed_path.open("rb") as fh:
        data = pickle.load(fh)
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = data
    if x_train_raw.shape[1] != NB_SENSOR_CHANNELS:
        raise SystemExit(f"Expected {NB_SENSOR_CHANNELS} channels, got {x_train_raw.shape[1]}")
    load_author_modules()
    from sliding_window import sliding_window

    def win(data_x, data_y, step):
        x = sliding_window(data_x, (SLIDING_WINDOW_LENGTH, data_x.shape[1]), (step, 1))
        y = np.asarray([[i[-1]] for i in sliding_window(data_y, SLIDING_WINDOW_LENGTH, step)])
        return x.astype(np.float32), y.reshape(len(y)).astype(np.uint8)

    x_test, y_test = win(x_test_raw, y_test_raw, TEST_SLIDING_WINDOW_STEP)
    x_train, y_train = win(x_train_raw, y_train_raw, SLIDING_WINDOW_STEP)
    return x_train, y_train, x_test, y_test, x_train_raw, y_train_raw.astype(np.uint8), x_test_raw, y_test_raw.astype(np.uint8)


def load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray, y_test_raw: np.ndarray, device: torch.device) -> dict:
    model.eval()
    test_pred = np.zeros_like(y_test_raw, dtype=np.uint8)
    start = SLIDING_WINDOW_LENGTH - 1
    with torch.no_grad():
        for inputs, targets in iter_full_batches(x_test, y_test):
            tensor = torch.from_numpy(inputs).unsqueeze(1).to(device)
            y_pred = model.predict(tensor).detach().cpu().numpy().astype(np.uint8)
            test_pred[start : start + targets.shape[0]] = y_pred
            start += targets.shape[0]
    labels = sorted(np.unique(y_test_raw).astype(int).tolist())
    mask = y_test_raw != 0
    non_null_labels = sorted(v for v in labels if v != 0)
    return {
        "accuracy": float(accuracy_score(y_test_raw, test_pred)),
        "f1_macro": float(f1_score(y_test_raw, test_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_test_raw, test_pred, average="weighted", labels=labels, zero_division=0)),
        "f1_macro_no_null": float(f1_score(y_test_raw[mask], test_pred[mask], average="macro", labels=non_null_labels, zero_division=0)),
        "accuracy_no_null": float(accuracy_score(y_test_raw[mask], test_pred[mask])),
        "evaluated_raw_samples": int(len(y_test_raw)),
        "test_windows": int(len(x_test)),
        "full_batch_predictions": int((len(x_test) // BATCH_SIZE) * BATCH_SIZE),
        "unfilled_tail_samples": int(len(y_test_raw) - (SLIDING_WINDOW_LENGTH - 1) - ((len(x_test) // BATCH_SIZE) * BATCH_SIZE)),
    }


def write_result(metrics: dict, args, runtime: float, state_path: Path, processed_path: Path, device: torch.device) -> Path:
    gap = metrics["f1_macro"] - PAPER_SCORE
    result = {
        "benchmark_id": "opportunity_a",
        "primary_score": metrics["f1_macro"],
        "metrics": metrics,
        "source": "author_checkpoint_official_archive",
        "protocol_status": "paper_protocol",
        "paper_reference_metric": "f1_macro",
        "paper_reference_score": PAPER_SCORE,
        "paper_reference_source": "Murahari & Ploetz 2018 Table 1; original archive checkpoint filename reports 0.7065222851937571.",
        "paper_reference_gap_signed": float(gap),
        "paper_reference_gap_abs": float(abs(gap)),
        "paper_reference_gap_threshold": GAP_THRESHOLD,
        "paper_reference_gap_status": "within_expected_range" if abs(gap) <= GAP_THRESHOLD else "large_gap_check_protocol",
        "runtime_seconds": round(runtime, 3),
        "device": str(device),
        "torch_version": torch.__version__,
        "state_dict_path": str(state_path),
        "processed_data_path": str(processed_path),
        "input_channels": NB_SENSOR_CHANNELS,
        "classes": NUM_CLASSES,
        "window_size_samples": SLIDING_WINDOW_LENGTH,
        "train_stride_samples": SLIDING_WINDOW_STEP,
        "test_stride_samples": TEST_SLIDING_WINDOW_STEP,
        "batch_size": BATCH_SIZE,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "lr_decay_from_checkpoint_name": args.lr_decay,
        "evaluation_note": "Replicates author sample-wise evaluation: full 100-sample batches fill raw test predictions from index window-1; unfilled prefix/tail remain class 0.",
    }
    if device.type == "cuda":
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(device)

    entry = {
        "id": "attention_model",
        "method_name": "Attention Model",
        "year": 2018,
        "method_family": "CNN/LSTM + Attention",
        "submission_type": "platform",
        "coverage_status": "author_checkpoint_paper_protocol",
        "summary": "Original Attention Model author checkpoint evaluated on the author's OPPORTUNITY preprocessing and sample-wise metric.",
        "implementation": "Author archive state_dict loaded into the original OPPORTUNITY architecture, with hard-coded paths adapted locally.",
        "har_scope": "OPPORTUNITY gesture recognition under the original paper protocol.",
        "datasets": [
            {
                "benchmark_id": "opportunity_a",
                "dataset": "OPPORTUNITY Activity Recognition",
                "activity_task": "18-class gesture recognition including Null, author 77-channel subset.",
                "reported_effect": "Author checkpoint f1_macro {:.4f} vs paper {:.4f}; gap {:+.4f}.".format(
                    metrics["f1_macro"], PAPER_SCORE, gap
                ),
                "protocol_note": "Author preprocessing uses 77 OPPORTUNITY features, 24-sample windows, train stride 12, test stride 1, and sample-wise macro-F1 including Null.",
            }
        ],
        "results": [result],
        "sources": ["https://arxiv.org/pdf/1805.07648", "https://www.dropbox.com/s/p7gt5sv0kgy3s16/AttentionModels.zip?dl=0"],
    }
    out_dir = RUNS_DIR / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    global AUTHOR_REPO
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", default=str(AUTHOR_REPO))
    parser.add_argument("--raw-root", default=str(OPPORTUNITY_ROOT))
    parser.add_argument("--processed-path", default=str(ROOT / "experiments" / "datasets" / "attention_model_author" / "oppChallenge_gestures.data"))
    parser.add_argument("--state-dict", default=str(AUTHOR_REPO / "0.7065222851937571..0.74..0.79..0.96_Opportunity"))
    parser.add_argument("--attention-dropout", type=float, default=0.74)
    parser.add_argument("--dropout", type=float, default=0.79)
    parser.add_argument("--lr-decay", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--run-tag", default="attention_model_author_checkpoint_opportunity_20260725")
    args = parser.parse_args(argv)

    AUTHOR_REPO = Path(args.author_repo)
    seed_all(args.seed)
    wants_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device(f"cuda:{args.gpu}" if wants_cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    processed_path = Path(args.processed_path)
    state_path = Path(args.state_dict)
    ensure_processed(processed_path, Path(args.raw_root), args.force_preprocess)
    x_train, y_train, x_test, y_test, x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataset(processed_path)
    print(f"author windows train={x_train.shape} test={x_test.shape} raw_test={x_test_raw.shape}", flush=True)
    model = AuthorAttentionRCNN(
        input_size=NB_SENSOR_CHANNELS * NUM_FILTERS,
        hidden_size=NUM_UNITS_LSTM,
        num_layers=NUM_LAYERS_LSTM,
        is_bidirectional=False,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
    ).to(device)
    state = load_state_dict(state_path, device)
    model.load_state_dict(state)
    start = time.time()
    metrics = evaluate(model, x_test, y_test, y_test_raw, device)
    runtime = time.time() - start
    out_path = write_result(metrics, args, runtime, state_path, processed_path, device)
    print(
        "Attention author checkpoint OPPORTUNITY: f1_macro={:.6f}, paper={:.6f}, gap={:+.6f}, wrote {}".format(
            metrics["f1_macro"], PAPER_SCORE, metrics["f1_macro"] - PAPER_SCORE, out_path
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
