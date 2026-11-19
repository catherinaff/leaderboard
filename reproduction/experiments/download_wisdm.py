"""Download and unpack WISDM v1.1 raw accelerometer data."""
from __future__ import annotations

import argparse
from pathlib import Path

from har_utils import DATASETS_DIR, download_file, extract_archive


URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
ARCHIVE = DATASETS_DIR / "wisdm_ar_latest.tar.gz"
OUT_DIR = DATASETS_DIR / "wisdm"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    download_file(URL, ARCHIVE, force=args.force)
    extract_archive(ARCHIVE, OUT_DIR, marker=OUT_DIR / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt", force=args.force)
    print(f"WISDM ready: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
