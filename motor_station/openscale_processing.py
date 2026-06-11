from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
KG_TO_NEWTONS = 9.80665


@dataclass
class RawSample:
    timestamp_unix: float
    timestamp_utc: str
    elapsed_s: float
    raw_text: str
    value_kg: Optional[float]
    unit: str
    is_numeric: bool


@dataclass
class CleanedSeries:
    timestamps_s: np.ndarray
    raw_values: np.ndarray
    despiked_values: np.ndarray
    filtered_values: np.ndarray
    outlier_mask: np.ndarray


def mass_to_force(values_kg: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(values_kg) * KG_TO_NEWTONS


def compute_impulse(timestamps_s: np.ndarray, force_n: np.ndarray, clamp_negative: bool = True) -> float:
    if timestamps_s.size == 0 or force_n.size == 0:
        return 0.0

    usable_force = np.asarray(force_n, dtype=float)
    if clamp_negative:
        usable_force = np.maximum(usable_force, 0.0)

    impulse = 0.0
    for index in range(1, min(timestamps_s.size, usable_force.size)):
        dt = float(timestamps_s[index] - timestamps_s[index - 1])
        if dt <= 0:
            continue
        impulse += 0.5 * (usable_force[index] + usable_force[index - 1]) * dt
    return float(impulse)


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def parse_sensor_line(line: str) -> tuple[Optional[float], str]:
    text = line.strip()
    if not text:
        return None, ""

    match = _FLOAT_RE.match(text)
    if not match:
        return None, ""

    try:
        value = float(match.group(0))
    except ValueError:
        return None, ""

    remainder = text[match.end():].strip(", \t")
    unit = ""
    if remainder:
        parts = [part.strip() for part in remainder.split(",") if part.strip()]
        if parts:
            unit = parts[0]

    return value, unit


def make_raw_sample(
    timestamp_unix: float,
    elapsed_s: float,
    raw_text: str,
) -> RawSample:
    value_kg, unit = parse_sensor_line(raw_text)
    return RawSample(
        timestamp_unix=timestamp_unix,
        timestamp_utc=utc_iso(timestamp_unix),
        elapsed_s=elapsed_s,
        raw_text=raw_text,
        value_kg=value_kg,
        unit=unit,
        is_numeric=value_kg is not None,
    )


def write_raw_session_header(
    file_obj,
    *,
    port: str,
    baudrate: int,
    started_at_unix: float,
) -> None:
    writer = csv.writer(file_obj)
    writer.writerow(["# OpenScale raw session"])
    writer.writerow(["# started_at_utc", utc_iso(started_at_unix)])
    writer.writerow(["# started_at_unix", f"{started_at_unix:.6f}"])
    writer.writerow(["# port", port])
    writer.writerow(["# baudrate", str(baudrate)])
    writer.writerow(
        [
            "timestamp_unix",
            "timestamp_utc",
            "elapsed_s",
            "raw_text",
            "value_kg",
            "unit",
            "is_numeric",
        ]
    )


def write_raw_sample(file_obj, sample: RawSample) -> None:
    writer = csv.writer(file_obj)
    writer.writerow(
        [
            f"{sample.timestamp_unix:.6f}",
            sample.timestamp_utc,
            f"{sample.elapsed_s:.6f}",
            sample.raw_text,
            "" if sample.value_kg is None else f"{sample.value_kg:.6f}",
            sample.unit,
            "1" if sample.is_numeric else "0",
        ]
    )


def _parse_structured_row(row: Sequence[str]) -> Optional[RawSample]:
    if not row:
        return None
    if row[0].startswith("#"):
        return None

    first = row[0].strip().lower()
    if first in {"timestamp_unix", "timestamp", "time"}:
        return None

    try:
        timestamp_unix = float(row[0])
    except (ValueError, IndexError):
        return None

    raw_text = ",".join(part for part in row[1:] if part is not None)
    timestamp_utc = ""

    if len(row) >= 7 and row[2].strip():
        timestamp_utc = row[1].strip()
        try:
            elapsed_s = float(row[2]) if row[2].strip() else 0.0
        except ValueError:
            elapsed_s = 0.0
        raw_text = row[3]
        value_kg = None
        if row[4].strip():
            try:
                value_kg = float(row[4])
            except ValueError:
                value_kg = None
        unit = row[5].strip()
        is_numeric = row[6].strip() in {"1", "true", "True"}
    elif len(row) == 2:
        timestamp_utc = utc_iso(timestamp_unix)
        elapsed_s = 0.0
        value_kg, unit = parse_sensor_line(raw_text)
        is_numeric = value_kg is not None
    else:
        timestamp_utc = row[1].strip() if len(row) > 1 else utc_iso(timestamp_unix)
        elapsed_s = 0.0
        value_kg = None
        unit = ""
        if len(row) > 4 and row[4].strip():
            try:
                value_kg = float(row[4])
            except ValueError:
                value_kg = None
            unit = row[5].strip() if len(row) > 5 else ""
        if value_kg is None:
            value_kg, unit = parse_sensor_line(raw_text)
        is_numeric = value_kg is not None

    if not timestamp_utc:
        timestamp_utc = utc_iso(timestamp_unix)

    return RawSample(
        timestamp_unix=timestamp_unix,
        timestamp_utc=timestamp_utc or utc_iso(timestamp_unix),
        elapsed_s=elapsed_s,
        raw_text=raw_text,
        value_kg=value_kg,
        unit=unit,
        is_numeric=is_numeric,
    )


def load_raw_log(path: str | Path) -> list[RawSample]:
    samples: list[RawSample] = []
    path = Path(path)

    with path.open("r", newline="") as file_obj:
        reader = csv.reader(file_obj)
        rows = list(reader)

    if not rows:
        return samples

    for row in rows:
        if not row:
            continue
        if row[0].startswith("#"):
            continue

        sample = _parse_structured_row(row)
        if sample is not None:
            samples.append(sample)

    return samples


def _median_absolute_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def hampel_filter(values: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window_size < 3:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    filtered = values.astype(float).copy()

    for index in range(values.size):
        left = max(0, index - half_window)
        right = min(values.size, index + half_window + 1)
        window = values[left:right]
        median = float(np.median(window))
        sigma = _median_absolute_deviation(window)
        if sigma == 0:
            continue
        if abs(values[index] - median) > n_sigmas * sigma:
            filtered[index] = median

    return filtered


def rolling_median(values: np.ndarray, window_size: int = 7) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window_size < 3:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed = values.astype(float).copy()

    for index in range(values.size):
        left = max(0, index - half_window)
        right = min(values.size, index + half_window + 1)
        smoothed[index] = float(np.median(values[left:right]))

    return smoothed


def low_pass_ema(values: np.ndarray, timestamps_s: np.ndarray, tau_s: float = 0.75) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if tau_s <= 0:
        return values.copy()

    smoothed = values.astype(float).copy()
    smoothed[0] = values[0]

    for index in range(1, values.size):
        dt = max(0.0, float(timestamps_s[index] - timestamps_s[index - 1]))
        alpha = dt / (tau_s + dt) if dt > 0 else 1.0
        smoothed[index] = smoothed[index - 1] + alpha * (values[index] - smoothed[index - 1])

    return smoothed


def build_clean_series(
    samples: Sequence[RawSample],
    *,
    window_size: int = 7,
    n_sigmas: float = 3.0,
    tau_s: float = 0.75,
) -> CleanedSeries:
    numeric_samples = [sample for sample in samples if sample.value_kg is not None]
    if not numeric_samples:
        empty = np.array([], dtype=float)
        return CleanedSeries(empty, empty, empty, empty, np.array([], dtype=bool))

    timestamps = np.array([sample.elapsed_s for sample in numeric_samples], dtype=float)
    raw_values = mass_to_force(np.array([float(sample.value_kg) for sample in numeric_samples], dtype=float))

    if timestamps.size > 1 and np.allclose(timestamps, 0.0):
        timestamps = np.array([sample.timestamp_unix for sample in numeric_samples], dtype=float)
        timestamps -= timestamps[0]

    despiked = hampel_filter(raw_values, window_size=window_size, n_sigmas=n_sigmas)
    local_baseline = rolling_median(despiked, window_size=window_size)
    filtered = low_pass_ema(local_baseline, timestamps, tau_s=tau_s)
    outlier_mask = np.abs(raw_values - despiked) > 0.0

    return CleanedSeries(
        timestamps_s=timestamps,
        raw_values=raw_values,
        despiked_values=despiked,
        filtered_values=filtered,
        outlier_mask=outlier_mask,
    )


def export_clean_csv(
    path: str | Path,
    samples: Sequence[RawSample],
    series: CleanedSeries,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["# Cleaned OpenScale data"])
        writer.writerow(
            [
                "timestamp_s",
                "raw_force_n",
                "despiked_force_n",
                "filtered_force_n",
                "is_outlier",
            ]
        )

        numeric_samples = [sample for sample in samples if sample.value_kg is not None]
        for index, _sample in enumerate(numeric_samples):
            writer.writerow(
                [
                    f"{series.timestamps_s[index]:.6f}",
                    f"{series.raw_values[index]:.6f}",
                    f"{series.despiked_values[index]:.6f}",
                    f"{series.filtered_values[index]:.6f}",
                    "1" if series.outlier_mask[index] else "0",
                ]
            )

    return path


def output_paths(input_path: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    return output_dir / f"{stem}_cleaned.csv", output_dir / f"{stem}_filtered.png"
