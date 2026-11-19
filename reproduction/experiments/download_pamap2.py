"""Download and unpack PAMAP2 from UCI."""
from __future__ import annotations

import argparse
import zipfile

from har_utils import DATASETS_DIR, download_file, extract_archive


URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
ARCHIVE = DATASETS_DIR / "pamap2.zip"
OUT_DIR = DATASETS_DIR / "pamap2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    download_file(URL, ARCHIVE, force=args.force)
    extract_archive(ARCHIVE, OUT_DIR, marker=OUT_DIR / "PAMAP2_Dataset" / "Protocol", force=args.force)
    nested = OUT_DIR / "PAMAP2_Dataset.zip"
    marker = OUT_DIR / "PAMAP2_Dataset" / "Protocol"
    if nested.is_file() and not marker.is_dir():
        print(f"Extracting nested archive {nested}")
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(OUT_DIR)
    print(f"PAMAP2 ready: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
