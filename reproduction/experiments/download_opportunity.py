"""Download and unpack OPPORTUNITY from UCI."""
from __future__ import annotations

import argparse

from har_utils import DATASETS_DIR, download_file, extract_archive


URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"
ARCHIVE = DATASETS_DIR / "opportunity.zip"
OUT_DIR = DATASETS_DIR / "opportunity"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    download_file(URL, ARCHIVE, force=args.force)
    extract_archive(ARCHIVE, OUT_DIR, marker=OUT_DIR / "OpportunityUCIDataset" / "dataset", force=args.force)
    print(f"OPPORTUNITY ready: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
