"""Download and unpack USC-HAD."""
from __future__ import annotations

import argparse

from har_utils import DATASETS_DIR, download_file, extract_archive


URL = "https://sipi.usc.edu/had/USC-HAD.zip"
ARCHIVE = DATASETS_DIR / "usc_had.zip"
OUT_DIR = DATASETS_DIR / "usc_had"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    download_file(URL, ARCHIVE, force=args.force)
    extract_archive(ARCHIVE, OUT_DIR, marker=OUT_DIR / "Subject1", force=args.force)
    print(f"USC-HAD ready: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
