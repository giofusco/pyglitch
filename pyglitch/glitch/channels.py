from __future__ import annotations

from dataclasses import dataclass

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.utils import (
    validate_8bit_value,
    validate_channel,
    validate_image,
    validate_uint8_image,
)
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class ReorderChannels(GlitchFilter):
    """Reorder the first three channels.

    Example:
        order=(2, 1, 0) swaps red and blue.
    """

    order: tuple[int, int, int]

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="order",
                kind=ParameterKind.ENUM,
                default=(0, 1, 2),
                enum_values=(
                    (0, 1, 2),
                    (0, 2, 1),
                    (1, 0, 2),
                    (1, 2, 0),
                    (2, 0, 1),
                    (2, 1, 0),
                ),
                label="Channel Order",
                description="Order used to write the first three output channels.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_image(image)

        if len(self.order) != 3:
            raise ValueError(f"Expected exactly 3 channel indices, got {self.order}.")

        for channel in self.order:
            validate_channel(image, channel)

        data = image.data.copy()
        data[:, :, 0] = image.data[:, :, self.order[0]]
        data[:, :, 1] = image.data[:, :, self.order[1]]
        data[:, :, 2] = image.data[:, :, self.order[2]]

        return image.with_data(data)


@dataclass(frozen=True)
class SetChannelValue(GlitchFilter):
    """Set every pixel in one channel to a constant 8-bit value."""

    channel: int
    value: int

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
                description="Channel index to overwrite.",
            ),
            ParameterSpec(
                name="value",
                kind=ParameterKind.INT,
                minimum=0,
                maximum=255,
                default=255,
                label="Value",
                description="8-bit value written into the selected channel.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)
        validate_channel(image, self.channel)
        validate_8bit_value(self.value, "value")

        data = image.data.copy()
        data[:, :, self.channel] = self.value

        return image.with_data(data)


@dataclass(frozen=True)
class SaturateChannel(SetChannelValue):
    """Set every pixel in one channel to 255."""

    value: int = 255

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
                description="Channel index to saturate.",
            ),
        )
