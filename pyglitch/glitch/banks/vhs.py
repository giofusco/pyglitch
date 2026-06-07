from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.filters import ConvolutionFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec, Parameterized
from pyglitch.glitch.pipeline import GlitchPipeline
from pyglitch.glitch.shift import ShiftChannel
from pyglitch.glitch.signal import SignalFlanger, SignalOutputMode, SignalRebuildMode, SignalScanMode
from pyglitch.glitch.types import Axis
from pyglitch.glitch.utils import validate_channel
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class VHS(GlitchFilter, Parameterized):
    """VHS-style macro filter.

    Combines:
    - horizontal channel shift
    - horizontal blur
    - signal flanger

    This is a bank/macro filter, not an atomic primitive.
    """

    blur_amount: int = 8
    shift_amount: int = 12
    shift_channel: int = 0
    flanger_delay: float = 0.0001
    flanger_rate: float = 0.75
    flanger_sample_rate: float = 48_100.0
    flanger_wet: float = 0.5
    flanger_scan_mode: SignalScanMode = SignalScanMode.ROW_MAJOR
    flanger_rebuild_mode: SignalRebuildMode = SignalRebuildMode.MATCH_SCAN
    flanger_output_mode: SignalOutputMode = SignalOutputMode.CLIP

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="blur_amount",
                kind=ParameterKind.INT,
                minimum=1,
                maximum=64,
                default=8,
                label="Blur Amount",
                description="Width of the horizontal box blur kernel.",
            ),
            ParameterSpec(
                name="shift_amount",
                kind=ParameterKind.INT,
                minimum=-256,
                maximum=256,
                default=12,
                label="Channel Shift",
                description="Horizontal pixel shift applied to one color channel.",
            ),
            ParameterSpec(
                name="shift_channel",
                kind=ParameterKind.INT,
                minimum=0,
                maximum=2,
                default=0,
                label="Shift Channel",
                description="Color channel affected by the horizontal shift.",
            ),
            ParameterSpec(
                name="flanger_delay",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=0.01,
                default=0.0001,
                label="Flanger Delay",
                description="Maximum flanger delay in seconds.",
            ),
            ParameterSpec(
                name="flanger_rate",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=10.0,
                default=0.75,
                label="Flanger Rate",
                description="Speed of the flanger modulation.",
            ),
            ParameterSpec(
                name="flanger_wet",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=0.5,
                label="Flanger Wet",
                description="Amount of delayed signal mixed into the image signal.",
            ),
            ParameterSpec(
                name="flanger_scan_mode",
                kind=ParameterKind.ENUM,
                default=SignalScanMode.ROW_MAJOR,
                enum_values=tuple(SignalScanMode),
                label="Flanger Scan Mode",
                description="How image pixels are read as a 1D signal.",
            ),
            ParameterSpec(
                name="flanger_rebuild_mode",
                kind=ParameterKind.ENUM,
                default=SignalRebuildMode.MATCH_SCAN,
                enum_values=tuple(SignalRebuildMode),
                label="Flanger Rebuild Mode",
                description="How the processed signal is written back to the image.",
            ),
            ParameterSpec(
                name="flanger_output_mode",
                kind=ParameterKind.ENUM,
                default=SignalOutputMode.CLIP,
                enum_values=tuple(SignalOutputMode),
                label="Flanger Output Mode",
                description="How processed float signal values are converted to uint8.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        if self.blur_amount <= 0:
            raise ValueError(f"blur_amount must be positive, got {self.blur_amount}.")

        validate_channel(image, self.shift_channel)

        kernel = np.ones((1, self.blur_amount), dtype=np.float32)
        kernel /= float(self.blur_amount)

        pipeline = GlitchPipeline(
            filters=(
                ShiftChannel(
                    channel=self.shift_channel,
                    offset=self.shift_amount,
                    axis=Axis.HORIZONTAL,
                ),
                ConvolutionFilter(kernel=kernel),
                SignalFlanger(
                    max_time_delay=self.flanger_delay,
                    rate=self.flanger_rate,
                    sample_rate=self.flanger_sample_rate,
                    dry=1.0,
                    wet=self.flanger_wet,
                    scan_mode=self.flanger_scan_mode,
                    rebuild_mode=self.flanger_rebuild_mode,
                    output_mode=self.flanger_output_mode,
                ),
            )
        )

        return pipeline.apply(image)
