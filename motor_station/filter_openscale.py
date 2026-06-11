from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openscale_processing import (
    compute_impulse,
    build_clean_series,
    export_clean_csv,
    load_raw_log,
    output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and plot OpenScale raw data")
    parser.add_argument("input", help="Raw OpenScale session file")
    parser.add_argument("--output-dir", default=None, help="Directory for cleaned outputs")
    parser.add_argument("--window-size", type=int, default=20, help="Hampel window size")
    parser.add_argument("--sigma", type=float, default=3.0, help="Outlier rejection threshold")
    parser.add_argument("--tau", type=float, default=0.01, help="Low-pass time constant in seconds")
    parser.add_argument("--no-show", action="store_true", help="Skip opening the interactive plot window")
    return parser.parse_args()


def render_plot(input_path: Path, csv_path: Path, png_path: Path, series) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(f"OpenScale filtering: {input_path.name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, alpha=0.25)

    ax.plot(series.timestamps_s, series.raw_values, color="#9ca3af", linewidth=1.4, marker="o", markersize=3, label="Raw")
    ax.plot(
        series.timestamps_s,
        series.despiked_values,
        color="#d97706",
        linewidth=1.8,
        label="Despiked",
    )
    ax.plot(
        series.timestamps_s,
        series.filtered_values,
        color="#2563eb",
        linewidth=2.4,
        label="Filtered",
    )

    if np.any(series.outlier_mask):
        ax.scatter(
            series.timestamps_s[series.outlier_mask],
            series.raw_values[series.outlier_mask],
            color="#dc2626",
            marker="x",
            s=50,
            label="Outliers",
            zorder=5,
        )

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    print(f"Saved cleaned CSV to: {csv_path}")
    print(f"Saved plot to: {png_path}")


def compute_average_frequency(timestamps_s: np.ndarray) -> float:
    if timestamps_s.size < 2:
        return 0.0

    deltas = np.diff(timestamps_s)
    valid_deltas = deltas[deltas > 0]
    if valid_deltas.size == 0:
        return 0.0

    return float(1.0 / np.mean(valid_deltas))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    samples = load_raw_log(input_path)
    if not samples:
        raise SystemExit(f"No usable samples found in {input_path}")

    series = build_clean_series(
        samples,
        window_size=args.window_size,
        n_sigmas=args.sigma,
        tau_s=args.tau,
    )
    if series.timestamps_s.size == 0:
        raise SystemExit(f"No numeric samples found in {input_path}")

    csv_path, png_path = output_paths(input_path, args.output_dir)
    export_clean_csv(csv_path, samples, series)
    render_plot(input_path, csv_path, png_path, series)

    total_impulse = compute_impulse(series.timestamps_s, series.filtered_values)
    print(f"Total impulse: {total_impulse:.6f} N*s")
    average_frequency = compute_average_frequency(series.timestamps_s)
    print(f"Average frequency: {average_frequency:.2f} Hz")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
