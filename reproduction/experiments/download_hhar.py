"""Download and unpack HHAR from UCI."""
from __future__ import annotations

import argparse
import zipfile

from har_utils import DATASETS_DIR, download_file, extract_archive


URL = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
ARCHIVE = DATASETS_DIR / "hhar.zip"
OUT_DIR = DATASETS_DIR / "hhar"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    download_file(URL, ARCHIVE, force=args.force)
    extract_archive(ARCHIVE, OUT_DIR, marker=OUT_DIR / "Activity recognition exp" / "Phones_accelerometer.csv", force=args.force)
    nested = OUT_DIR / "Activity recognition exp.zip"
    marker = OUT_DIR / "Activity recognition exp" / "Phones_accelerometer.csv"
    if nested.is_file() and not marker.is_file():
        print(f"Extracting nested archive {nested}")
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(OUT_DIR)
    print(f"HHAR ready: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
