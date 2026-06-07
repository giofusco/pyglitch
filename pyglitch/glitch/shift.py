from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.types import Axis
from pyglitch.glitch.utils import validate_channel, validate_image
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class ShiftImage(GlitchFilter):
    """Circularly shift the whole image along one axis."""

    offset: int
    axis: Axis = Axis.HORIZONTAL

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="offset",
                kind=ParameterKind.INT,
                minimum=-2048,
                maximum=2048,
                default=0,
                label="Offset",
                description="Pixel offset applied to the whole image.",
            ),
            ParameterSpec(
                name="axis",
                kind=ParameterKind.ENUM,
                default=Axis.HORIZONTAL,
                enum_values=tuple(Axis),
                label="Axis",
                description="Axis along which the image is shifted.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_image(image)

        axis_index = 1 if self.axis is Axis.HORIZONTAL else 0
        data = np.roll(image.data, shift=self.offset, axis=axis_index)

        return image.with_data(data)


@dataclass(frozen=True)
class ShiftChannel(GlitchFilter):
    """Circularly shift a single channel along one axis."""

    channel: int
    offset: int
    axis: Axis = Axis.HORIZONTAL

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="channel",
                kind=ParameterKind.INT,
                minimum=0,
                maximum=2,
                default=0,
                label="Channel",
                description="Channel index to shift.",
            ),
            ParameterSpec(
                name="offset",
                kind=ParameterKind.INT,
                minimum=-2048,
                maximum=2048,
                default=0,
                label="Offset",
                description="Pixel offset applied to the selected channel.",
            ),
            ParameterSpec(
                name="axis",
                kind=ParameterKind.ENUM,
                default=Axis.HORIZONTAL,
                enum_values=tuple(Axis),
                label="Axis",
                description="Axis along which the channel is shifted.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_image(image)
        validate_channel(image, self.channel)

        data = image.data.copy()
        axis_index = 1 if self.axis is Axis.HORIZONTAL else 0

        data[:, :, self.channel] = np.roll(
            image.data[:, :, self.channel],
            shift=self.offset,
            axis=axis_index,
        )

        return image.with_data(data)


@dataclass(frozen=True)
class SineShiftRows(GlitchFilter):
    """Shift each row horizontally using a sine-wave displacement."""

    amplitude: float
    frequency: float
    phase: float = 0.0

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="amplitude",
                kind=ParameterKind.FLOAT,
                minimum=-2048.0,
                maximum=2048.0,
                default=16.0,
                label="Amplitude",
                description="Maximum horizontal displacement in pixels.",
            ),
            ParameterSpec(
                name="frequency",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=128.0,
                default=3.0,
                label="Frequency",
                description="Number of sine cycles across the image height.",
            ),
            ParameterSpec(
                name="phase",
                kind=ParameterKind.FLOAT,
                minimum=-6.283185307179586,
                maximum=6.283185307179586,
                default=0.0,
                label="Phase",
                description="Sine phase offset in radians.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_image(image)

        if image.height == 0:
            return image.copy()

        data = image.data.copy()
        row_positions = np.arange(image.height)

        offsets = np.round(
            self.amplitude
            * np.sin(
                self.phase
                + self.frequency * 2.0 * np.pi * row_positions / image.height
            )
        ).astype(int)

        for row_index, offset in enumerate(offsets):
            data[row_index, :, :] = np.roll(
                image.data[row_index, :, :],
                shift=offset,
                axis=0,
            )

        return image.with_data(data)


@dataclass(frozen=True)
class SineShiftColumns(GlitchFilter):
    """Shift each column vertically using a sine-wave displacement."""

    amplitude: float
    frequency: float
    phase: float = 0.0

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return SineShiftRows.parameter_specs()

    def apply(self, image: GlitchImage) -> GlitchImage:
        rotated = image.rotated_right()
        shifted = SineShiftRows(
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=self.phase,
        ).apply(rotated)

        return shifted.rotated_left()
