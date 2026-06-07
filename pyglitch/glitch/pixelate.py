from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.types import PixelateOperator
from pyglitch.glitch.utils import validate_uint8_image
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class Pixelate(GlitchFilter):
    """Pixelate an image by replacing each block with one representative color."""

    block_height: int = 5
    block_width: int | None = None
    operator: PixelateOperator = PixelateOperator.MEAN

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="block_height",
                kind=ParameterKind.INT,
                minimum=1,
                maximum=512,
                default=5,
                label="Block Height",
                description="Pixelation block height in pixels.",
            ),
            ParameterSpec(
                name="block_width",
                kind=ParameterKind.INT,
                minimum=1,
                maximum=512,
                default=5,
                label="Block Width",
                description="Pixelation block width in pixels. Defaults to block_height when None.",
            ),
            ParameterSpec(
                name="operator",
                kind=ParameterKind.ENUM,
                default=PixelateOperator.MEAN,
                enum_values=tuple(PixelateOperator),
                label="Operator",
                description="Aggregation method used to compute each block color.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)

        if self.block_height <= 0:
            raise ValueError(
                f"block_height must be positive, got {self.block_height}."
            )

        block_width = self.block_width
        if block_width is None:
            block_width = self.block_height

        if block_width <= 0:
            raise ValueError(f"block_width must be positive, got {block_width}.")

        data = image.data.copy()

        for top in range(0, image.height, self.block_height):
            for left in range(0, image.width, block_width):
                bottom = min(top + self.block_height, image.height)
                right = min(left + block_width, image.width)

                block = image.data[top:bottom, left:right, :]
                color = _compute_pixelate_color(block, self.operator)

                data[top:bottom, left:right, :] = color

        return image.with_data(data)


def _compute_pixelate_color(
    block: np.ndarray,
    operator: PixelateOperator,
) -> np.ndarray:
    if operator is PixelateOperator.MEAN:
        return np.mean(block, axis=(0, 1))

    if operator is PixelateOperator.MEDIAN:
        return np.median(block, axis=(0, 1))

    if operator is PixelateOperator.MAXIMUM:
        return np.max(block, axis=(0, 1))

    if operator is PixelateOperator.MINIMUM:
        return np.min(block, axis=(0, 1))

    raise ValueError(f"Unsupported pixelate operator: {operator}.")
