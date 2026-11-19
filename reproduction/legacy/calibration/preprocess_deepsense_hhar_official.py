"""DeepSense HHAR preprocessing using the official HHAR-Data-Process logic.

The original author code is Python 2 and writes every raw row with a separate
open/close. This version keeps the same grouping, acc/gyro pairing, FFT feature
construction, augmentation, and one-user-out split, while making it resumable and
parameterized for select=a/b/c.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import interp1d


GT_TYPES = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
TIME_SCALE = 100000000.0
TIME_LABEL = "Creation_Time"
SEPCTURAL_SAMPLES = 10
FFT_SPAN = 5.0
SAMP_SPAN = 20.0
TIME_NOISE_VAR = 0.2
ACC_NOISE_VAR = 0.5
GYRO_NOISE_VAR = 0.2
AUG_NUM = 10
WIDE = 20
WIDE_SCALE_FACTOR = 4
INPUT_FEATURE = SEPCTURAL_SAMPLES * 6 * 2


class HandleCache:
    def __init__(self, max_open: int = 256) -> None:
        self.max_open = max_open
        self.handles: OrderedDict[Path, object] = OrderedDict()

    def get(self, path: Path):
        handle = self.handles.pop(path, None)
        if handle is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", buffering=1024 * 1024)
        self.handles[path] = handle
        if len(self.handles) > self.max_open:
            _, old = self.handles.popitem(last=False)
            old.close()
        return handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def split_raw(raw_dir: Path, split_dir: Path, manifest_path: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    cache = HandleCache()
    counts: dict[str, int] = {}
    raw_files = [
        raw_dir / "Phones_accelerometer.csv",
        raw_dir / "Phones_gyroscope.csv",
        raw_dir / "Watch_accelerometer.csv",
        raw_dir / "Watch_gyroscope.csv",
    ]
    try:
        for raw_file in raw_files:
            sensor = "accelerometer" if "accelerometer" in raw_file.name else "gyroscope"
            print(f"[split] {raw_file}", flush=True)
            with raw_file.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader, 1):
                    gt = row.get("gt", "")
                    if gt not in GT_TYPES:
                        continue
                    user = row["User"]
                    device = row["Device"]
                    key = f"{user}-{device}-{gt}"
                    out = split_dir / f"{key}-{sensor}.tsv"
                    cache.get(out).write(
                        "\t".join(
                            [
                                row["Creation_Time"],
                                row["x"],
                                row["y"],
                                row["z"],
                            ]
                        )
                        + "\n"
                    )
                    counts[str(out.name)] = counts.get(str(out.name), 0) + 1
                    if idx % 1_000_000 == 0:
                        print(f"[split] {raw_file.name}: {idx} rows", flush=True)
    finally:
        cache.close()
    manifest_path.write_text(json.dumps({"counts": counts}, indent=2), encoding="utf-8")
    print(f"[split] wrote {len(counts)} split files", flush=True)


def _read_sensor(path: Path) -> list[tuple[float, list[float]]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            t, x, y, z = line.rstrip("\n").split("\t")
            rows.append((float(t) / TIME_SCALE, [float(x), float(y), float(z)]))
    return rows


def pair_acc_gyro(split_dir: Path, pair_dir: Path, manifest_path: Path) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    acc_files = sorted(split_dir.glob("*-accelerometer.tsv"))
    counts = {}
    for idx, acc_path in enumerate(acc_files, 1):
        key = acc_path.name[: -len("-accelerometer.tsv")]
        gyro_path = split_dir / f"{key}-gyroscope.tsv"
        if not gyro_path.is_file():
            continue
        out_path = pair_dir / key
        if out_path.is_file() and out_path.stat().st_size > 0:
            counts[key] = sum(1 for _ in out_path.open())
            continue
        acc = _read_sensor(acc_path)
        gyro = _read_sensor(gyro_path)
        i = 0
        j = 0
        wrote = 0
        with out_path.open("w", buffering=1024 * 1024) as out:
            while i < len(acc) and j < len(gyro):
                t1, a = acc[i]
                t2, g = gyro[j]
                if abs(t1 - t2) < 0.1:
                    elem = {
                        "Time": 0.5 * (t1 + t2),
                        "Accelerometer": a,
                        "Gyroscope": g,
                    }
                    out.write(repr(elem) + "\n")
                    i += 1
                    j += 1
                    wrote += 1
                elif t1 - t2 >= 0.1:
                    j += 1
                else:
                    i += 1
        counts[key] = wrote
        if idx % 25 == 0:
            print(f"[pair] {idx}/{len(acc_files)} files; last {key} -> {wrote}", flush=True)
    manifest_path.write_text(json.dumps({"counts": counts}, indent=2), encoding="utf-8")
    print(f"[pair] wrote {len(counts)} paired files", flush=True)


def _load_pair_file(path: Path):
    with path.open() as handle:
        for line in handle:
            yield eval(line, {"__builtins__": {}})


def _sequence_features(pair_dir: Path, name_dev: str, gt: str, select: str, rng: np.random.Generator):
    cur_type = GT_TYPES.index(gt)
    cur_data_sep = []
    path = pair_dir / f"{name_dev}-{gt}"
    if not path.is_file():
        return []

    cur_data = []
    cur_acc_list = []
    cur_gyro_list = []
    cur_time_list = []
    last_time = -1.0
    start_time = -1.0

    for elem in _load_pair_file(path):
        cur_time = float(elem["Time"])
        if start_time < 0:
            start_time = cur_time
        if abs(cur_time - start_time) > FFT_SPAN:
            if cur_acc_list:
                if cur_time_list[-1] < FFT_SPAN:
                    cur_acc_list.append(deepcopy(cur_acc_list[-1]))
                    cur_gyro_list.append(deepcopy(cur_gyro_list[-1]))
                    cur_time_list.append(FFT_SPAN)
                cur_acc_org = np.asarray(cur_acc_list, dtype=np.float64).T
                cur_gyro_org = np.asarray(cur_gyro_list, dtype=np.float64).T
                cur_time_org = np.asarray(cur_time_list, dtype=np.float64)
                cur_aug_num = 1 if name_dev[0] == select else AUG_NUM

                for aug_idx in range(cur_aug_num):
                    if aug_idx == 0:
                        cur_acc = cur_acc_org.copy()
                        cur_gyro = cur_gyro_org.copy()
                        cur_times = cur_time_org.copy()
                    else:
                        cur_acc = cur_acc_org + rng.normal(0.0, ACC_NOISE_VAR, cur_acc_org.shape)
                        cur_gyro = cur_gyro_org + rng.normal(0.0, GYRO_NOISE_VAR, cur_gyro_org.shape)
                        cur_times = cur_time_org + rng.normal(0.0, TIME_NOISE_VAR, cur_time_org.shape)

                    cur_times = np.sort(cur_times)
                    if cur_times[-1] < FFT_SPAN:
                        cur_times[-1] = FFT_SPAN
                    if cur_times[0] > 0.0:
                        cur_times[0] = 0.0

                    try:
                        acc_interp = interp1d(cur_times, cur_acc)
                        gyro_interp = interp1d(cur_times, cur_gyro)
                    except Exception:
                        continue
                    interp_time = np.linspace(0.0, FFT_SPAN, SEPCTURAL_SAMPLES)
                    acc_fft = fft(acc_interp(interp_time)).T
                    gyro_fft = fft(gyro_interp(interp_time)).T
                    cur_sen_data = []
                    for acc_elem in acc_fft:
                        for axis_elem in acc_elem:
                            cur_sen_data.append(axis_elem.real)
                            cur_sen_data.append(axis_elem.imag)
                    for gyro_elem in gyro_fft:
                        for axis_elem in gyro_elem:
                            cur_sen_data.append(axis_elem.real)
                            cur_sen_data.append(axis_elem.imag)

                    if len(cur_data) < AUG_NUM:
                        cur_data.append([deepcopy(cur_sen_data)])
                    elif start_time - last_time >= SAMP_SPAN and aug_idx == 0:
                        for cur_aug_data in cur_data:
                            cur_data_sep.append(deepcopy(cur_aug_data))
                        cur_data = [[deepcopy(cur_sen_data)]]
                    else:
                        cur_data[aug_idx].append(deepcopy(cur_sen_data))

                last_time = start_time + cur_time_list[-1]
                start_time = -1.0
                cur_acc_list = []
                cur_gyro_list = []
                cur_time_list = []
        if start_time < 0:
            start_time = cur_time
        offset = cur_time - start_time
        if offset not in cur_time_list:
            cur_acc_list.append(deepcopy(elem["Accelerometer"]))
            cur_gyro_list.append(deepcopy(elem["Gyroscope"]))
            cur_time_list.append(offset)

    examples = []
    label = [0.0] * len(GT_TYPES)
    label[cur_type] = 1.0
    for sep_data in cur_data_sep:
        sta_idx = 0
        while sta_idx < len(sep_data):
            end_idx = min(sta_idx + WIDE, len(sep_data))
            if end_idx - sta_idx < 5:
                break
            window = deepcopy(sep_data[sta_idx:end_idx])
            while len(window) < WIDE:
                window.append([0.0] * INPUT_FEATURE)
            examples.append((window, label))
            sta_idx += int(WIDE / WIDE_SCALE_FACTOR)
    return examples


def make_sep(pair_dir: Path, out_root: Path, select: str, seed: int | None, manifest_path: Path) -> None:
    # The Python 2 official script calls np.random.normal directly. RandomState
    # keeps the same legacy generator family while still allowing reproducible reruns.
    rng = np.random.RandomState(seed) if seed is not None else np.random
    name_devs = []
    for path in sorted(pair_dir.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.name.startswith("#"):
            continue
        elems = path.name.split("-")
        name_dev = "-".join(elems[:-1])
        if name_dev not in name_devs:
            name_devs.append(name_dev)

    out_dir = out_root / f"sepHARData_{select}"
    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    eval_count = 0
    train_support = [0] * len(GT_TYPES)
    eval_support = [0] * len(GT_TYPES)

    for nd_idx, name_dev in enumerate(name_devs, 1):
        target_eval = name_dev[0] == select
        for gt in GT_TYPES:
            examples = _sequence_features(pair_dir, name_dev, gt, select, rng)
            label_idx = GT_TYPES.index(gt)
            for window, label in examples:
                arr = np.asarray(window, dtype=np.float64).reshape(-1).tolist() + label
                line = ",".join(str(float(x)) for x in arr) + "\n"
                if target_eval:
                    (eval_dir / f"eval_{eval_count}.csv").write_text(line)
                    eval_count += 1
                    eval_support[label_idx] += 1
                else:
                    (train_dir / f"train_{train_count}.csv").write_text(line)
                    train_count += 1
                    train_support[label_idx] += 1
        if nd_idx % 5 == 0:
            print(
                f"[sep {select}] {nd_idx}/{len(name_devs)} devices train={train_count} eval={eval_count}",
                flush=True,
            )

    manifest = {
        "select": select,
        "train_count": train_count,
        "eval_count": eval_count,
        "train_support": train_support,
        "eval_support": eval_support,
        "name_devs": name_devs,
        "seed": seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[sep {select}] done {manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--select", choices=["a", "b", "c"], action="append", required=True)
    parser.add_argument("--stage", choices=["all", "split", "pair", "sep"], default="all")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    split_dir = args.work_dir / "Dataset_Alg_SourceDevice_fast"
    pair_dir = args.work_dir / f"Dataset_AccGry_SourceDevice-{TIME_LABEL}-avgTime"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "split"):
        split_raw(args.raw_dir, split_dir, args.work_dir / "split_manifest.json")
    if args.stage in ("all", "pair"):
        pair_acc_gyro(split_dir, pair_dir, args.work_dir / "pair_manifest.json")
    if args.stage in ("all", "sep"):
        for select in args.select:
            make_sep(
                pair_dir,
                args.out_root,
                select,
                args.seed,
                args.work_dir / f"sepHARData_{select}_manifest.json",
            )
    print(f"[done] runtime_seconds={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
