"""Inspect LIMU-BERT prepared array shapes for TPN reproduction."""
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "experiments" / "repos" / "LIMU-BERT-Public" / "dataset"


def main() -> None:
    for dataset in ["hhar", "unimib"]:
        d = DATASET_ROOT / dataset
        print(f"## {dataset} {d.exists()} {d}")
        for p in sorted(d.glob("*20_120*.npy")):
            arr = np.load(p, mmap_mode="r")
            print(p.name, arr.shape, arr.dtype)
        for p in sorted(d.glob("*label*.npy")):
            arr = np.load(p, mmap_mode="r")
            print(p.name, arr.shape, arr.dtype)


if __name__ == "__main__":
    main()
