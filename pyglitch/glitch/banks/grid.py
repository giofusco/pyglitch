from __future__ import annotations

from dataclasses import dataclass

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec, Parameterized
from pyglitch.glitch.signal import SignalOutputMode, SignalRebuildMode, SignalScanMode, SignalTremolo
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class Grid(GlitchFilter, Parameterized):
    """Grid-like macro filter based on horizontal and vertical tremolo passes."""

    thickness: float = 1.5
    darkness: float = 0.25
    sample_rate: float = 48_100.0
    output_mode: SignalOutputMode = SignalOutputMode.CLIP

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="thickness",
                kind=ParameterKind.FLOAT,
                minimum=0.01,
                maximum=40.0,
                default=1.5,
                label="Grid Thickness",
                description="Tremolo frequency used for both grid directions.",
            ),
            ParameterSpec(
                name="darkness",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=0.25,
                label="Grid Darkness",
                description="Depth of the tremolo modulation.",
            ),
            ParameterSpec(
                name="sample_rate",
                kind=ParameterKind.FLOAT,
                minimum=1.0,
                maximum=192_000.0,
                default=48_100.0,
                label="Sample Rate",
                description="Virtual sample rate used by the signal tremolo.",
            ),
            ParameterSpec(
                name="output_mode",
                kind=ParameterKind.ENUM,
                default=SignalOutputMode.CLIP,
                enum_values=tuple(SignalOutputMode),
                label="Output Mode",
                description="How processed values are converted to uint8.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        horizontal = SignalTremolo(
        frequency=self.thickness,
        depth=self.darkness,
        sample_rate=self.sample_rate,
        scan_mode=SignalScanMode.ROW_MAJOR,
        rebuild_mode=SignalRebuildMode.MATCH_SCAN,
        output_mode=self.output_mode,
        ).apply(image)

        vertical = SignalTremolo(
            frequency=self.thickness,
            depth=self.darkness,
            sample_rate=self.sample_rate,
            scan_mode=SignalScanMode.COLUMN_MAJOR,
            rebuild_mode=SignalRebuildMode.MATCH_SCAN,
            output_mode=self.output_mode,
        ).apply(horizontal)

        return vertical
