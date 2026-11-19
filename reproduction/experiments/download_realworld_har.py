"""Download RealWorld-HAR from the official Mannheim distribution.

The original full archive is about 3.7GB and was very slow from the legacy
HTTP host. The official dataset page also exposes per-subject CSV zip files.
For this leaderboard protocol we only need accelerometer and gyroscope streams,
so the default path downloads those official CSV zips and extracts them into a
layout that `preprocess_realworld_har.py` can consume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

from har_utils import DATASETS_DIR, download_file, write_json


FULL_ARCHIVE_URL = "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"
CSV_BASE_URL = "https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Projects/sensor/realworld"
SUBJECTS = tuple(range(1, 16))
ACTIVITIES = (
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
)
SENSORS = ("acc", "gyr")

ARCHIVE = DATASETS_DIR / "realworld_har.zip"
CSV_ZIPS_DIR = DATASETS_DIR / "realworld_har_csv_zips"
RAW_DIR = DATASETS_DIR / "realworld_har"
BLOCKER = DATASETS_DIR / "realworld_har_download_blocker.json"
MANIFEST = DATASETS_DIR / "realworld_har_download_manifest.json"


def remote_size(url: str) -> int | None:
    try:
        with urlopen(Request(url, method="HEAD", headers={"User-Agent": "IoTBench-HAR/0.1"}), timeout=20) as resp:
            return int(resp.headers.get("content-length") or 0)
    except Exception:
        return None


def csv_url(subject: int, sensor: str, activity: str) -> str:
    return f"{CSV_BASE_URL}/s{subject}/data/{sensor}_{activity}_csv.zip"


def download_one_csv_zip(subject: int, sensor: str, activity: str, *, force: bool = False) -> dict:
    url = csv_url(subject, sensor, activity)
    dest = CSV_ZIPS_DIR / f"s{subject:02d}" / f"{sensor}_{activity}_csv.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return {
            "subject": subject,
            "sensor": sensor,
            "activity": activity,
            "url": url,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "status": "already_present",
        }

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    req = Request(url, headers={"User-Agent": "IoTBench-HAR/0.1"})
    with urlopen(req, timeout=90) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return {
        "subject": subject,
        "sensor": sensor,
        "activity": activity,
        "url": url,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "status": "downloaded",
    }


def extract_csv_zips(records: list[dict], *, force: bool = False) -> list[dict]:
    extracted = []
    for record in records:
        if record.get("status") not in {"downloaded", "already_present"}:
            continue
        subject = int(record["subject"])
        sensor = str(record["sensor"])
        activity = str(record["activity"])
        archive = Path(record["path"])
        out_dir = RAW_DIR / f"s{subject:02d}" / activity / sensor
        marker = out_dir / ".extracted"
        if marker.exists() and not force:
            extracted.append({**record, "extract_dir": str(out_dir), "extract_status": "already_extracted"})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        marker.write_text("ok\n", encoding="utf-8")
        extracted.append({**record, "extract_dir": str(out_dir), "extract_status": "extracted"})
    return extracted


def download_csv_distribution(args: argparse.Namespace) -> int:
    jobs = [(subject, sensor, activity) for subject in SUBJECTS for activity in ACTIVITIES for sensor in SENSORS]
    if not args.confirm_large and not any(CSV_ZIPS_DIR.rglob("*.zip")):
        write_json(
            BLOCKER,
            {
                "dataset": "RealWorld-HAR",
                "status": "blocked_official_csv_download_requires_confirmation",
                "source": "official_subject_activity_sensor_csv_zips",
                "source_base_url": CSV_BASE_URL,
                "expected_archives": len(jobs),
                "notes": "The canonical protocol needs the official per-subject accelerometer and gyroscope CSV zips. Re-run with --confirm-large to download them.",
            },
        )
        print(f"Blocked by download guard. Wrote {BLOCKER}")
        return 0

    records: list[dict] = []
    failures: list[dict] = []
    print(f"Downloading {len(jobs)} official RealWorld-HAR CSV zip files with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(download_one_csv_zip, subject, sensor, activity, force=args.force): (subject, sensor, activity)
            for subject, sensor, activity in jobs
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            subject, sensor, activity = future_map[future]
            try:
                record = future.result()
                records.append(record)
                if index == 1 or index % 20 == 0 or index == len(jobs):
                    total_mb = sum(item.get("size_bytes", 0) for item in records) / (1024 * 1024)
                    print(f"  {index}/{len(jobs)} complete, {total_mb:.1f} MB downloaded/present")
            except Exception as exc:
                failure = {
                    "subject": subject,
                    "sensor": sensor,
                    "activity": activity,
                    "url": csv_url(subject, sensor, activity),
                    "status": "failed",
                    "error": repr(exc),
                }
                failures.append(failure)
                print(f"  failed s{subject:02d} {sensor} {activity}: {exc}")

    extracted = extract_csv_zips(records, force=args.force_extract) if not args.no_extract else records
    status = "complete" if not failures and len(records) == len(jobs) else "incomplete"
    manifest = {
        "dataset": "RealWorld-HAR",
        "status": status,
        "source": "official_subject_activity_sensor_csv_zips",
        "source_base_url": CSV_BASE_URL,
        "expected_archives": len(jobs),
        "completed_archives": len(records),
        "failed_archives": len(failures),
        "total_size_bytes": sum(item.get("size_bytes", 0) for item in records),
        "zips_dir": str(CSV_ZIPS_DIR),
        "raw_dir": str(RAW_DIR),
        "records": sorted(extracted, key=lambda item: (item["subject"], item["activity"], item["sensor"])),
        "failures": sorted(failures, key=lambda item: (item["subject"], item["activity"], item["sensor"])),
    }
    write_json(MANIFEST, manifest)
    if status == "complete" and BLOCKER.exists():
        BLOCKER.unlink()
    print(json.dumps({k: manifest[k] for k in ("dataset", "status", "completed_archives", "failed_archives", "total_size_bytes")}, indent=2))
    if failures:
        print(f"Wrote incomplete manifest: {MANIFEST}")
        return 1
    print(f"Wrote manifest: {MANIFEST}")
    return 0


def download_full_archive(args: argparse.Namespace) -> int:
    size = remote_size(FULL_ARCHIVE_URL)
    if not args.confirm_large and not ARCHIVE.is_file():
        write_json(
            BLOCKER,
            {
                "dataset": "RealWorld-HAR",
                "status": "blocked_large_download_requires_confirmation",
                "url": FULL_ARCHIVE_URL,
                "remote_size_bytes": size,
                "notes": "The public full archive is about 3.7GB. Re-run with --confirm-large --source full-archive to download before preprocessing.",
            },
        )
        print(f"Blocked by large download guard. Wrote {BLOCKER}")
        return 0

    download_file(FULL_ARCHIVE_URL, ARCHIVE, force=args.force)
    manifest = {
        "dataset": "RealWorld-HAR",
        "status": "complete",
        "source": "full_archive",
        "url": FULL_ARCHIVE_URL,
        "archive": str(ARCHIVE),
        "size_bytes": ARCHIVE.stat().st_size,
    }
    write_json(MANIFEST, manifest)
    if BLOCKER.exists():
        BLOCKER.unlink()
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["official-csv", "full-archive"],
        default="official-csv",
        help="Use official per-subject acc/gyr CSV zips, or the legacy full archive.",
    )
    parser.add_argument("--confirm-large", action="store_true", help="Actually download the official RealWorld-HAR files.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.source == "full-archive":
        return download_full_archive(args)
    return download_csv_distribution(args)


if __name__ == "__main__":
    raise SystemExit(main())
