"""Preprocess RealWorld-HAR into canonical aligned IMU windows.

The public RealWorld archive is large, so this script is written to be safe:
if the archive is absent it writes a blocker JSON and exits without fabricating
windows. Once `download_realworld_har.py --confirm-large` has downloaded the
archive, this script extracts it and builds the canonical 50 Hz LOSO windows.
"""
from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from har_utils import DATASETS_DIR, PROCESSED_DIR, extract_archive, load_benchmark, save_windows, write_json


BENCHMARK_ID = "realworld_har"
ARCHIVE = DATASETS_DIR / "realworld_har.zip"
CSV_ZIPS_DIR = DATASETS_DIR / "realworld_har_csv_zips"
RAW_DIR = DATASETS_DIR / "realworld_har"
EXTRACT_MARKER = RAW_DIR / ".extracted"
BLOCKER = DATASETS_DIR / "realworld_har_preprocess_blocker.json"
OUT = PROCESSED_DIR / "realworld_har_windows.npz"

POSITIONS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
SENSORS = ["acc", "gyr"]
AXES = ["x", "y", "z"]

ACTIVITY_ALIASES = {
    "climbingdown": "Climbing Down",
    "climbing_down": "Climbing Down",
    "downstairs": "Climbing Down",
    "climbingup": "Climbing Up",
    "climbing_up": "Climbing Up",
    "upstairs": "Climbing Up",
    "jumping": "Jumping",
    "lying": "Lying",
    "running": "Running",
    "jogging": "Running",
    "sitting": "Sitting",
    "standing": "Standing",
    "walking": "Walking",
}
SENSOR_ALIASES = {
    "acc": "acc",
    "accel": "acc",
    "accelerometer": "acc",
    "gyr": "gyr",
    "gyro": "gyr",
    "gyroscope": "gyr",
}
POSITION_ALIASES = {
    "chest": "chest",
    "forearm": "forearm",
    "lowerarm": "forearm",
    "head": "head",
    "shin": "shin",
    "lowerleg": "shin",
    "thigh": "thigh",
    "upperleg": "thigh",
    "upperarm": "upperarm",
    "upper_arm": "upperarm",
    "waist": "waist",
    "hip": "waist",
}


def write_blocker(status: str, notes: str, **extra) -> None:
    write_json(
        BLOCKER,
        {
            "dataset": "RealWorld-HAR",
            "benchmark_id": BENCHMARK_ID,
            "status": status,
            "archive": str(ARCHIVE),
            "notes": notes,
            **extra,
        },
    )
    print(f"Blocked: {notes}. Wrote {BLOCKER}")


def extract_nested_zips(root: Path, *, force: bool = False) -> None:
    for archive in list(root.rglob("*.zip")):
        out_dir = archive.with_suffix("")
        marker = out_dir / ".extracted"
        if marker.exists() and not force:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting nested archive {archive} -> {out_dir}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        marker.write_text("ok\n", encoding="utf-8")


def target_dir_for_csv_zip(archive: Path) -> Path:
    subject_dir = archive.relative_to(CSV_ZIPS_DIR).parts[0]
    parts = archive.stem.split("_")
    sensor = parts[0]
    activity = "_".join(parts[1:-1]) if parts[-1] == "csv" else "_".join(parts[1:])
    return RAW_DIR / subject_dir / activity / sensor


def extract_csv_zip_distribution(*, force: bool = False) -> int:
    if force and RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    count = 0
    for archive in sorted(CSV_ZIPS_DIR.rglob("*.zip")):
        out_dir = target_dir_for_csv_zip(archive)
        marker = out_dir / ".extracted"
        if marker.exists() and not force:
            count += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        marker.write_text("ok\n", encoding="utf-8")
        count += 1
    return count


def prepare_raw_dir(*, force_extract: bool = False) -> tuple[bool, str]:
    if ARCHIVE.is_file():
        extract_archive(ARCHIVE, RAW_DIR, marker=EXTRACT_MARKER, force=force_extract)
        EXTRACT_MARKER.write_text("ok\n", encoding="utf-8")
        extract_nested_zips(RAW_DIR, force=force_extract)
        return True, "full_archive"

    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.csv")) and not force_extract:
        return True, "official_csv_zips_pre_extracted"

    extracted = extract_csv_zip_distribution(force=force_extract)
    if extracted:
        return True, "official_csv_zips"

    if RAW_DIR.exists() and any(RAW_DIR.rglob("*.csv")):
        return True, "pre_extracted_csv"

    return False, "missing_raw_data"


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def subject_from_path(path: Path) -> int | None:
    for part in path.parts:
        lower = part.lower()
        match = re.search(r"(?:proband|subject|subj|user|s)[_-]?0*([0-9]{1,3})", lower)
        if match:
            return int(match.group(1))
    return None


def first_alias(text: str, aliases: dict[str, str]) -> str | None:
    lowered = text.lower()
    compact = normalize_token(text)
    for key, value in aliases.items():
        if key in lowered or normalize_token(key) in compact:
            return value
    return None


def file_metadata(path: Path) -> tuple[int, str, str, str] | None:
    text = "_".join(path.with_suffix("").parts[-4:])
    subject = subject_from_path(path)
    activity = first_alias(text, ACTIVITY_ALIASES)
    sensor = first_alias(text, SENSOR_ALIASES)
    position = first_alias(text, POSITION_ALIASES)
    if subject is None or activity is None or sensor is None or position is None:
        return None
    return subject, activity, sensor, position


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            cols.append(str(col))
    return cols


def pick_axis_columns(df: pd.DataFrame) -> list[str] | None:
    lowered = {str(col).lower(): str(col) for col in df.columns}
    candidates = []
    for axis in AXES:
        exact = [
            lowered.get(axis),
            lowered.get(f"attr_{axis}"),
            lowered.get(f"accelerometer_{axis}"),
            lowered.get(f"gyroscope_{axis}"),
        ]
        col = next((item for item in exact if item is not None), None)
        if col is not None:
            candidates.append(col)
    if len(candidates) == 3:
        return candidates

    numeric = numeric_columns(df)
    non_time = [col for col in numeric if "time" not in col.lower() and "timestamp" not in col.lower()]
    if len(non_time) >= 3:
        return non_time[-3:]
    return None


def pick_time_column(df: pd.DataFrame, axis_cols: list[str]) -> str | None:
    for col in df.columns:
        lower = str(col).lower()
        if "time" in lower or "timestamp" in lower:
            return str(col)
    numeric = [col for col in numeric_columns(df) if col not in axis_cols]
    return numeric[0] if numeric else None


def normalize_time(values: np.ndarray, fallback_len: int, sample_hz: float) -> np.ndarray:
    if values.size == 0 or not np.isfinite(values).any():
        return np.arange(fallback_len, dtype=np.float64) / sample_hz
    values = values.astype(np.float64)
    finite = np.sort(values[np.isfinite(values)])
    diffs = np.diff(finite)
    diffs = diffs[diffs > 0]
    median_diff = float(np.median(diffs)) if diffs.size else 1.0 / sample_hz
    raw_max = float(np.nanmax(values))

    if raw_max > 1e16 or median_diff > 1e6:
        scale = 1e9
    elif raw_max > 1e13 or median_diff > 1e3:
        scale = 1e6
    elif raw_max > 1e10 or median_diff > 1:
        scale = 1e3
    else:
        scale = 1.0

    return (values - np.nanmin(values)) / scale


def read_stream(path: Path, sample_hz: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable CSV {path}: {exc}")
            return None
    if df.empty:
        return None
    axis_cols = pick_axis_columns(df)
    if axis_cols is None:
        print(f"Skipping CSV without x/y/z columns: {path}")
        return None
    xyz = df[axis_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    time_col = pick_time_column(df, axis_cols)
    if time_col is None:
        times = np.arange(len(xyz), dtype=np.float64) / sample_hz
    else:
        times = normalize_time(pd.to_numeric(df[time_col], errors="coerce").to_numpy(), len(xyz), sample_hz)
    keep = np.isfinite(times) & np.isfinite(xyz).any(axis=1)
    times = times[keep]
    xyz = xyz[keep]
    if len(times) < 2:
        return None
    order = np.argsort(times)
    times = times[order]
    xyz = xyz[order]
    unique, index = np.unique(times, return_index=True)
    return unique, xyz[index]


def interpolate_stream(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    cols = []
    for axis in range(values.shape[1]):
        col = values[:, axis]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            cols.append(np.zeros_like(grid, dtype=np.float32))
        else:
            cols.append(np.interp(grid, times[finite], col[finite]).astype(np.float32))
    return np.stack(cols, axis=1)


def collect_streams(raw_dir: Path, sample_hz: float) -> dict[tuple[int, str], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    candidates: dict[tuple[int, str], dict[tuple[str, str], Path]] = {}
    streams: dict[tuple[int, str], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]] = {}
    for path in sorted(raw_dir.rglob("*.csv")):
        meta = file_metadata(path)
        if meta is None:
            continue
        subject, activity, sensor, position = meta
        group = candidates.setdefault((subject, activity), {})
        key = (sensor, position)
        previous = group.get(key)
        if previous is None or path.stat().st_size > previous.stat().st_size:
            group[key] = path

    selected_count = sum(len(group) for group in candidates.values())
    print(f"Selected {selected_count} RealWorld stream CSVs from {len(candidates)} subject/activity groups")
    for group_key, group_paths in sorted(candidates.items()):
        group = streams.setdefault(group_key, {})
        for key, path in sorted(group_paths.items()):
            stream = read_stream(path, sample_hz)
            if stream is not None:
                group[key] = stream
    return streams


def build_segments(streams, sample_hz: float, label_to_id: dict[str, int]):
    segments = []
    skipped = []
    channel_names = [f"{position}_{sensor}_{axis}" for position in POSITIONS for sensor in SENSORS for axis in AXES]
    required = [(sensor, position) for position in POSITIONS for sensor in SENSORS]
    for (subject, activity), group in sorted(streams.items()):
        missing = [f"{sensor}_{position}" for sensor, position in required if (sensor, position) not in group]
        if missing:
            skipped.append({"subject": subject, "activity": activity, "reason": "missing_streams", "missing": missing})
            continue
        start = max(group[key][0][0] for key in required)
        end = min(group[key][0][-1] for key in required)
        if end <= start:
            skipped.append({"subject": subject, "activity": activity, "reason": "no_common_time_range"})
            continue
        grid = np.arange(start, end, 1.0 / sample_hz, dtype=np.float64)
        max_stream_len = max(len(group[key][0]) for key in required)
        if len(grid) > max_stream_len * 3:
            skipped.append(
                {
                    "subject": subject,
                    "activity": activity,
                    "reason": "implausible_time_grid",
                    "grid_samples": int(len(grid)),
                    "max_stream_samples": int(max_stream_len),
                }
            )
            continue
        if len(grid) < 2:
            skipped.append({"subject": subject, "activity": activity, "reason": "too_short_after_alignment"})
            continue
        aligned = []
        for position in POSITIONS:
            for sensor in SENSORS:
                times, values = group[(sensor, position)]
                aligned.append(interpolate_stream(times, values, grid))
        values = np.concatenate(aligned, axis=1)
        segments.append((subject, label_to_id[activity], values.astype(np.float32)))
    return segments, skipped, channel_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args(argv)

    bench = load_benchmark(BENCHMARK_ID)
    protocol = bench["protocol"]
    sample_hz = float(protocol["sampling_rate_hz"])
    window_size = int(protocol["window_size_samples"])
    stride = int(protocol["stride_samples"])

    prepared, source = prepare_raw_dir(force_extract=args.force_extract)
    if not prepared:
        write_blocker(
            "blocked_download_requires_confirmation",
            "RealWorld-HAR source files are not present. Run experiments/download_realworld_har.py --confirm-large before preprocessing.",
            required_protocol="benchmarks/realworld_har.json",
            accepted_sources=["official_csv_zips", "full_archive"],
        )
        return 0

    streams = collect_streams(RAW_DIR, sample_hz)
    if not streams:
        write_blocker(
            "blocked_unrecognized_archive_layout",
            "No RealWorld-HAR CSV files matching subject/activity/sensor/position naming conventions were found.",
        )
        return 0

    label_names = list(bench["classes"])
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    segments, skipped, channel_names = build_segments(streams, sample_hz, label_to_id)
    if not segments:
        write_blocker(
            "blocked_no_complete_segments",
            "No subject/activity segment had the full required acc/gyr streams for all seven positions.",
            skipped_segments=skipped[:50],
            discovered_groups=len(streams),
        )
        return 0

    from har_utils import windows_from_segments

    x, y, subjects = windows_from_segments(segments, window_size=window_size, stride=stride)
    save_windows(
        OUT,
        x=x,
        y=y,
        subjects=subjects,
        benchmark_id=BENCHMARK_ID,
        label_names=label_names,
        channel_names=channel_names,
        meta={
            "source": source,
            "source_archive": str(ARCHIVE) if ARCHIVE.is_file() else None,
            "source_csv_zips_dir": str(CSV_ZIPS_DIR) if CSV_ZIPS_DIR.exists() else None,
            "sample_hz": sample_hz,
            "window_size": window_size,
            "stride": stride,
            "subjects": sorted({int(s) for s in subjects.tolist()}),
            "positions": POSITIONS,
            "sensors": SENSORS,
            "segments_used": len(segments),
            "segments_skipped": skipped,
            "protocol_note": protocol.get("note", ""),
        },
    )
    if BLOCKER.exists():
        BLOCKER.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
