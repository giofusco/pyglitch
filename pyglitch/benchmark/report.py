from __future__ import annotations

import csv
from pathlib import Path

from pyglitch.benchmark.runner import BenchmarkResult


def print_summary(
    results: list[BenchmarkResult],
    target_fps: tuple[float, ...] = (60.0, 30.0, 15.0),
) -> None:
    if not results:
        print("No benchmark results.")
        return

    print("")
    print("RESULTS")
    print("-" * 128)

    header_targets = " ".join(f"{int(fps):>4d}fps" for fps in target_fps)

    print(
        f"{'Filter':28s} "
        f"{'Parameter':18s} "
        f"{'Value':>12s} "
        f"{'Shape':>18s} "
        f"{'Mean':>12s} "
        f"{'Median':>12s} "
        f"{'FPS':>10s} "
        f"{header_targets}"
    )

    print("-" * 128)

    for result in results:
        fps_flags = " ".join(
            f"{'Y' if result.meets_fps(fps) else 'N':>7s}"
            for fps in target_fps
        )

        print(
            f"{result.name:28s} "
            f"{result.parameter_name:18s} "
            f"{str(result.parameter_value):>12s} "
            f"{str(result.image_shape):>18s} "
            f"{result.mean_ms:9.3f} ms "
            f"{result.median_ms:9.3f} ms "
            f"{result.fps:10.2f} "
            f"{fps_flags}"
        )


def write_csv(
    results: list[BenchmarkResult],
    filename: str | Path,
) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "parameter_name",
                "parameter_value",
                "image_shape",
                "image_pixels",
                "iterations",
                "warmup",
                "mean_seconds",
                "median_seconds",
                "min_seconds",
                "max_seconds",
                "std_seconds",
                "fps",
                "mean_ms",
                "median_ms",
                "meets_60fps",
                "meets_30fps",
                "meets_15fps",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "parameter_name": result.parameter_name,
                    "parameter_value": result.parameter_value,
                    "image_shape": result.image_shape,
                    "image_pixels": result.image_pixels,
                    "iterations": result.iterations,
                    "warmup": result.warmup,
                    "mean_seconds": result.mean_seconds,
                    "median_seconds": result.median_seconds,
                    "min_seconds": result.min_seconds,
                    "max_seconds": result.max_seconds,
                    "std_seconds": result.std_seconds,
                    "fps": result.fps,
                    "mean_ms": result.mean_ms,
                    "median_ms": result.median_ms,
                    "meets_60fps": result.meets_fps(60.0),
                    "meets_30fps": result.meets_fps(30.0),
                    "meets_15fps": result.meets_fps(15.0),
                }
            )
