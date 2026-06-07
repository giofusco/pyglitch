from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import mean, median, stdev
import time

from pyglitch.glitch.base import GlitchFilter
from pyglitch.image.glitch_image import GlitchImage


FilterFactory = Callable[[object], GlitchFilter]


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    parameter_name: str
    parameter_value: object
    image_shape: tuple[int, ...]
    image_pixels: int
    iterations: int
    warmup: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    std_seconds: float

    @property
    def fps(self) -> float:
        if self.mean_seconds <= 0:
            return float("inf")

        return 1.0 / self.mean_seconds

    @property
    def mean_ms(self) -> float:
        return self.mean_seconds * 1000.0

    @property
    def median_ms(self) -> float:
        return self.median_seconds * 1000.0

    def meets_fps(self, target_fps: float) -> bool:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}.")

        return self.mean_seconds <= 1.0 / target_fps


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    parameter_name: str
    values: tuple[object, ...]
    build_filter: FilterFactory


def benchmark_filter(
    filter_: GlitchFilter,
    image: GlitchImage,
    iterations: int = 10,
    warmup: int = 2,
) -> tuple[float, ...]:
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}.")

    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")

    for _ in range(warmup):
        _ = filter_.apply(image)

    times: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        _ = filter_.apply(image)
        times.append(time.perf_counter() - start)

    return tuple(times)


def run_case(
    case: BenchmarkCase,
    image: GlitchImage,
    iterations: int = 10,
    warmup: int = 2,
    progress: bool = True,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for value in case.values:
        filter_ = case.build_filter(value)

        if progress:
            print(
                f"{case.name} [{case.parameter_name}={value}] ",
                end="",
                flush=True,
            )

        times = benchmark_filter(
            filter_=filter_,
            image=image,
            iterations=iterations,
            warmup=warmup,
        )

        if progress:
            print("█" * iterations, flush=True)

        std_seconds = stdev(times) if len(times) > 1 else 0.0

        image_pixels = int(image.height * image.width)

        results.append(
            BenchmarkResult(
                name=case.name,
                parameter_name=case.parameter_name,
                parameter_value=value,
                image_shape=tuple(image.shape),
                image_pixels=image_pixels,
                iterations=iterations,
                warmup=warmup,
                mean_seconds=mean(times),
                median_seconds=median(times),
                min_seconds=min(times),
                max_seconds=max(times),
                std_seconds=std_seconds,
            )
        )

    return results


def run_cases(
    cases: Iterable[BenchmarkCase],
    image: GlitchImage,
    iterations: int = 10,
    warmup: int = 2,
    progress: bool = True,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for case in cases:
        results.extend(
            run_case(
                case=case,
                image=image,
                iterations=iterations,
                warmup=warmup,
                progress=progress,
            )
        )

    return results


def run_cases_for_widths(
    cases: Iterable[BenchmarkCase],
    images_by_label: dict[str, GlitchImage],
    iterations: int = 10,
    warmup: int = 2,
    progress: bool = True,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for label, image in images_by_label.items():
        if progress:
            print("")
            print(f"=== Image: {label} {image.shape} ===")

        results.extend(
            run_cases(
                cases=cases,
                image=image,
                iterations=iterations,
                warmup=warmup,
                progress=progress,
            )
        )

    return results
