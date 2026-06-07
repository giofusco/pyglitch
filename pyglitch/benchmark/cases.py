from __future__ import annotations

from pyglitch.benchmark.runner import BenchmarkCase
from pyglitch.glitch import (
    Axis,
    PixelSortBrightSegments,
    Pixelate,
    Posterize,
    ShiftChannel,
    ShiftImage,
)
from pyglitch.glitch.signal import (
    SignalFlanger,
    SignalReverb,
    SignalTremolo,
    SignalWahWah,
)


def default_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            name="ShiftChannel",
            parameter_name="offset",
            values=tuple(range(-128, 129, 32)),
            build_filter=lambda value: ShiftChannel(
                channel=0,
                offset=int(value),
                axis=Axis.HORIZONTAL,
            ),
        ),
        BenchmarkCase(
            name="ShiftImage",
            parameter_name="offset",
            values=tuple(range(-128, 129, 32)),
            build_filter=lambda value: ShiftImage(
                offset=int(value),
                axis=Axis.HORIZONTAL,
            ),
        ),
        BenchmarkCase(
            name="Posterize",
            parameter_name="bins",
            values=(2, 4, 8, 16, 32, 64),
            build_filter=lambda value: Posterize(
                bins=int(value),
                normalize=False,
            ),
        ),
        BenchmarkCase(
            name="Pixelate",
            parameter_name="block_height",
            values=(2, 4, 8, 16, 32, 64),
            build_filter=lambda value: Pixelate(
                block_height=int(value),
            ),
        ),
        BenchmarkCase(
            name="PixelSortBrightSegments",
            parameter_name="threshold",
            values=(32, 64, 96, 128, 160, 192, 224),
            build_filter=lambda value: PixelSortBrightSegments(
                red_threshold=int(value),
                green_threshold=int(value),
                blue_threshold=int(value),
                strict=False,
            ),
        ),
        BenchmarkCase(
            name="SignalTremolo",
            parameter_name="frequency",
            values=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
            build_filter=lambda value: SignalTremolo(
                frequency=float(value),
                depth=0.5,
            ),
        ),
        BenchmarkCase(
            name="SignalFlanger",
            parameter_name="max_time_delay",
            values=(0.0, 0.0005, 0.001, 0.003, 0.006, 0.01),
            build_filter=lambda value: SignalFlanger(
                max_time_delay=float(value),
                rate=0.75,
                wet=0.7,
            ),
        ),
        BenchmarkCase(
            name="SignalReverb",
            parameter_name="delay_pixels",
            values=(16, 64, 128, 512, 2048, 8192),
            build_filter=lambda value: SignalReverb(
                delay_pixels=int(value),
                decay=0.5,
                feedback=True,
            ),
        ),
        BenchmarkCase(
            name="SignalWahWah",
            parameter_name="sweep_frequency",
            values=(20.0, 100.0, 500.0, 1000.0, 2000.0),
            build_filter=lambda value: SignalWahWah(
                sweep_frequency=float(value),
            ),
        ),
    )
