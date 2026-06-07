from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyglitch.benchmark import (
    default_benchmark_cases,
    print_summary,
    run_cases_for_widths,
    write_csv,
)
from pyglitch.benchmark.image_utils import make_preview_images, timestamped_report_path
from pyglitch.image import GlitchImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pyglitch filters across preview sizes."
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Measured iterations per parameter value.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations before measuring.",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[320, 640, 960],
        help="Preview widths to benchmark.",
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also benchmark the full-resolution image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Defaults to reports/filter_benchmark_<timestamp>.csv.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Input image does not exist: {image_path}", file=sys.stderr)
        return 1

    print("")
    print("** pyglitch filter benchmark **")
    print(f"Loading image: {image_path}")

    image = GlitchImage.load(image_path)

    images_by_label = make_preview_images(
        image=image,
        widths=tuple(args.widths),
        include_full=args.include_full,
    )

    cases = default_benchmark_cases()

    results = run_cases_for_widths(
        cases=cases,
        images_by_label=images_by_label,
        iterations=args.iterations,
        warmup=args.warmup,
        progress=not args.no_progress,
    )

    print_summary(results)

    output_path = Path(args.output) if args.output else timestamped_report_path()
    write_csv(results, output_path)

    print("")
    print(f"CSV report written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
