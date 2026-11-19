"""Evaluate the official DeepConvLSTM Lasagne checkpoint on OPPORTUNITY gestures.

This runner intentionally does not produce a canonical leaderboard row. The
official repository publishes a Python2/Lasagne notebook and a pretrained
OPPORTUNITY gesture checkpoint, but no all-dataset training entrypoint. The
result written here is therefore marked as a dataset variant.
"""
from __future__ import print_function

import argparse
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
from lib2to3.refactor import RefactoringTool, get_fixers_from_package


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = ROOT / "experiments" / "repos" / "DeepConvLSTM_official"
OFFICIAL_REPO_URL = "https://github.com/STRCWearlab/DeepConvLSTM"
OFFICIAL_COMMIT = "ea25e34da3a48b151a97bd3502802db547523b7f"
OPPORTUNITY_ROOT = ROOT / "experiments" / "datasets" / "opportunity"
RUN_DIR = ROOT / "experiments" / "runs" / "opportunity_a_deepconvlstm_official_checkpoint"

NB_SENSOR_CHANNELS = 113
NUM_CLASSES = 18
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12
FINAL_SEQUENCE_LENGTH = 8
BATCH_SIZE = 100
NUM_FILTERS = 64
FILTER_SIZE = 5
NUM_UNITS_LSTM = 128


def load_py2_module(path, module_name):
    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    source = path.read_text(encoding="utf-8")
    if not source.endswith("\n"):
        source += "\n"
    converted = str(tool.refactor_string(source, str(path)))
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(converted, str(path), "exec"), module.__dict__)
    return module


def load_official_preprocess():
    sys.path.insert(0, str(OFFICIAL_REPO))
    preprocess = load_py2_module(OFFICIAL_REPO / "preprocess_data.py", "official_deepconvlstm_preprocess")
    sliding = load_py2_module(OFFICIAL_REPO / "sliding_window.py", "official_deepconvlstm_sliding_window")
    return preprocess, sliding.sliding_window


def load_official_split(dataset_root, label):
    preprocess, _sliding_window = load_official_preprocess()
    data_x = np.empty((0, NB_SENSOR_CHANNELS), dtype=np.float32)
    data_y = np.empty((0,), dtype=np.uint8)
    for relative_name in preprocess.OPPORTUNITY_DATA_FILES:
        path = dataset_root / relative_name
        if not path.is_file():
            raise SystemExit("Missing OPPORTUNITY file required by official DeepConvLSTM: {}".format(path))
        print("official preprocess {}".format(relative_name), flush=True)
        raw = np.loadtxt(str(path))
        x_part, y_part = preprocess.process_dataset_file(raw, label)
        data_x = np.vstack((data_x, x_part.astype(np.float32)))
        data_y = np.concatenate((data_y, y_part.astype(np.uint8)))

    split_at = 557963
    if len(data_x) <= split_at:
        raise SystemExit("Official OPPORTUNITY split has no test rows; got {}".format(len(data_x)))
    return data_x[:split_at], data_y[:split_at], data_x[split_at:], data_y[split_at:]


def official_sliding_windows(data_x, data_y):
    _preprocess, sliding_window = load_official_preprocess()
    x = sliding_window(data_x, (SLIDING_WINDOW_LENGTH, data_x.shape[1]), (SLIDING_WINDOW_STEP, 1))
    y = np.asarray([[i[-1]] for i in sliding_window(data_y, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)])
    x = x.astype(np.float32).reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    y = y.reshape(len(y)).astype(np.uint8)
    return x, y


def build_official_network():
    os.environ.setdefault("THEANO_FLAGS", "device=cpu,floatX=float32")
    import lasagne

    net = {}
    net["input"] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    net["conv1/5x1"] = lasagne.layers.Conv2DLayer(net["input"], NUM_FILTERS, (FILTER_SIZE, 1))
    net["conv2/5x1"] = lasagne.layers.Conv2DLayer(net["conv1/5x1"], NUM_FILTERS, (FILTER_SIZE, 1))
    net["conv3/5x1"] = lasagne.layers.Conv2DLayer(net["conv2/5x1"], NUM_FILTERS, (FILTER_SIZE, 1))
    net["conv4/5x1"] = lasagne.layers.Conv2DLayer(net["conv3/5x1"], NUM_FILTERS, (FILTER_SIZE, 1))
    net["shuff"] = lasagne.layers.DimshuffleLayer(net["conv4/5x1"], (0, 2, 1, 3))
    net["lstm1"] = lasagne.layers.LSTMLayer(net["shuff"], NUM_UNITS_LSTM)
    net["lstm2"] = lasagne.layers.LSTMLayer(net["lstm1"], NUM_UNITS_LSTM)
    net["shp1"] = lasagne.layers.ReshapeLayer(net["lstm2"], (-1, NUM_UNITS_LSTM))
    net["prob"] = lasagne.layers.DenseLayer(net["shp1"], NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
    net["shp2"] = lasagne.layers.ReshapeLayer(net["prob"], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
    net["output"] = lasagne.layers.SliceLayer(net["shp2"], -1, 1)
    weights_path = OFFICIAL_REPO / "weights" / "DeepConvLSTM_oppChallenge_gestures.pkl"
    with weights_path.open("rb") as handle:
        params = pickle.load(handle, encoding="latin1")
    lasagne.layers.set_all_param_values(net["output"], params)
    return net


def iterate_minibatches(inputs, targets, batchsize, limit_batches=None):
    emitted = 0
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if limit_batches is not None and emitted >= limit_batches:
            break
        excerpt = slice(start_idx, start_idx + batchsize)
        emitted += 1
        yield inputs[excerpt], targets[excerpt]


def f1_scores(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_label = []
    weights = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        support = float(np.sum(y_true == label))
        if tp == 0.0 and (fp > 0.0 or fn > 0.0):
            f1 = 0.0
        elif support == 0.0 and fp == 0.0:
            continue
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label.append(f1)
        weights.append(support)
    macro = float(np.mean(per_label)) if per_label else 0.0
    weighted = float(np.average(per_label, weights=weights)) if sum(weights) else 0.0
    return macro, weighted


def write_result(metrics, runtime, evaluated_windows, dropped_windows):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": "deepconvlstm",
        "method_name": "DeepConvLSTM",
        "year": 2016,
        "method_family": "CNN + LSTM",
        "submission_type": "platform",
        "coverage_status": "official_checkpoint_dataset_variant",
        "summary": "Official DeepConvLSTM Lasagne checkpoint evaluation on the repository's OPPORTUNITY gesture protocol.",
        "implementation": "Runs the official Lasagne/Theano network from the DeepConvLSTM notebook with the released OPPORTUNITY gesture checkpoint.",
        "har_scope": "Wearable multimodal sequence classification with four temporal convolution layers followed by two LSTM layers.",
        "datasets": [
            {
                "benchmark_id": "opportunity_a",
                "dataset": "OPPORTUNITY Activity Recognition",
                "activity_task": "Official repository gesture task with 18 labels including Null/unannotated windows.",
                "subjects": "Official challenge-style file split from the repository notebook; not leaderboard LOSO.",
                "device_placement": "113 OPPORTUNITY challenge sensor channels.",
                "reported_effect": "Local official checkpoint evaluation f1_weighted {:.4f}.".format(metrics["f1_weighted"]),
                "protocol_note": "Dataset variant: official checkpoint uses 24-sample windows, 12-sample stride, challenge-style split, and an 18-class gesture label space including Null. The leaderboard OPPORTUNITY-A protocol uses ADL-only LOSO, 30-sample windows, 15-sample stride, and 17 evaluated gesture classes."
            }
        ],
        "results": [
            {
                "benchmark_id": "opportunity_a",
                "primary_score": metrics["f1_macro"],
                "metrics": metrics,
                "source": "official_checkpoint_evaluation",
                "protocol_status": "dataset_variant",
                "runtime_seconds": round(runtime, 3),
                "official_code_used": True,
                "official_repo": OFFICIAL_REPO_URL,
                "official_commit": OFFICIAL_COMMIT,
                "official_model_source": "DeepConvLSTM.ipynb Lasagne network and weights/DeepConvLSTM_oppChallenge_gestures.pkl",
                "official_reported_metric": "f1_weighted",
                "window_size_samples": SLIDING_WINDOW_LENGTH,
                "stride_samples": SLIDING_WINDOW_STEP,
                "batch_size": BATCH_SIZE,
                "evaluated_windows": int(evaluated_windows),
                "dropped_incomplete_batch_windows": int(dropped_windows),
                "classes": NUM_CLASSES,
                "device": "cpu",
                "theano_flags": os.environ.get("THEANO_FLAGS", ""),
            }
        ],
        "sources": [
            OFFICIAL_REPO_URL,
            "https://www.mdpi.com/1424-8220/16/1/115",
            "https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition",
        ],
    }
    out = RUN_DIR / "results.json"
    out.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(out), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(OPPORTUNITY_ROOT))
    parser.add_argument("--label", default="gestures", choices=["gestures"])
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args(argv)

    start = time.time()
    _x_train, _y_train, x_test_raw, y_test_raw = load_official_split(Path(args.dataset_root), args.label)
    x_test, y_test = official_sliding_windows(x_test_raw, y_test_raw)
    print("official windows test={} labels={}".format(x_test.shape, y_test.shape), flush=True)

    net = build_official_network()
    import theano
    import theano.tensor as T
    import lasagne

    prediction = lasagne.layers.get_output(net["output"], deterministic=True)
    test_fn = theano.function([net["input"].input_var], [T.argmax(prediction, axis=1)])

    preds = []
    trues = []
    for inputs, targets in iterate_minibatches(x_test, y_test, BATCH_SIZE, args.limit_batches):
        y_pred, = test_fn(inputs)
        preds.append(y_pred)
        trues.append(targets)
    if not preds:
        raise SystemExit("No full minibatches available for official evaluation.")
    y_pred = np.concatenate(preds).astype(np.uint8)
    y_true = np.concatenate(trues).astype(np.uint8)
    labels = list(range(NUM_CLASSES))
    f1_macro, f1_weighted = f1_scores(y_true, y_pred, labels)
    accuracy = float(np.mean(y_true == y_pred))
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    runtime = time.time() - start
    dropped = len(x_test) - len(y_true)
    write_result(metrics, runtime, len(y_true), dropped)
    print("DeepConvLSTM official checkpoint: f1_weighted={:.4f}, f1_macro={:.4f}, accuracy={:.4f}, runtime={:.1f}s".format(f1_weighted, f1_macro, accuracy, runtime), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
