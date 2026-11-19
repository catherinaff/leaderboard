"""Download and unpack UCI-HAR into experiments/datasets."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "experiments" / "datasets"
URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_PATH = DATASETS_DIR / "uci_har.zip"
OUT_DIR = DATASETS_DIR / "uci_har"


def main() -> int:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    marker = OUT_DIR / "UCI HAR Dataset" / "train" / "X_train.txt"
    if marker.is_file():
        print(f"UCI-HAR already present: {OUT_DIR}")
        return 0

    print(f"Downloading {URL}")
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=context) as resp, ZIP_PATH.open("wb") as fh:
        shutil.copyfileobj(resp, fh)

    tmp_dir = DATASETS_DIR / "_uci_har_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(tmp_dir)

    nested_zip = tmp_dir / "UCI HAR Dataset.zip"
    if nested_zip.is_file():
        with zipfile.ZipFile(nested_zip) as zf:
            zf.extractall(tmp_dir)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    extracted = tmp_dir / "UCI HAR Dataset"
    if not extracted.is_dir():
        raise RuntimeError("Unexpected UCI-HAR archive layout")
    shutil.move(str(extracted), str(OUT_DIR / "UCI HAR Dataset"))
    shutil.rmtree(tmp_dir)
    print(f"UCI-HAR ready: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
