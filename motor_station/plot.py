from __future__ import annotations

import argparse
from pathlib import Path

from filter_openscale import main as filter_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for OpenScale plotting")
    parser.add_argument("input", nargs="?", default="plot.txt", help="Raw OpenScale log to analyze")
    parser.add_argument("--output-dir", default=None, help="Directory for cleaned outputs")
    parser.add_argument("--window-size", type=int, default=7, help="Hampel window size")
    parser.add_argument("--sigma", type=float, default=3.0, help="Outlier rejection threshold")
    parser.add_argument("--tau", type=float, default=0.75, help="Low-pass time constant in seconds")
    parser.add_argument("--no-show", action="store_true", help="Skip opening the interactive plot window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        args.input,
        "--window-size",
        str(args.window_size),
        "--sigma",
        str(args.sigma),
        "--tau",
        str(args.tau),
    ]
    if args.output_dir is not None:
        argv += ["--output-dir", args.output_dir]
    if args.no_show:
        argv.append("--no-show")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        filter_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
