from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.utils import validate_uint8_image
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class Epoplectic(GlitchFilter):
    """Directional brightness erosion.

    For each pixel, compare it with a shifted neighbor. If the current pixel
    is brighter, replace it with the shifted neighbor.

    Positive x_shift reads from the right.
    Positive y_shift reads from below.
    """

    x_shift: int
    y_shift: int
    feedback: bool = False

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="x_shift",
                kind=ParameterKind.INT,
                minimum=-512,
                maximum=512,
                default=8,
                label="X Shift",
                description="Horizontal neighbor offset used for brightness erosion.",
            ),
            ParameterSpec(
                name="y_shift",
                kind=ParameterKind.INT,
                minimum=-512,
                maximum=512,
                default=8,
                label="Y Shift",
                description="Vertical neighbor offset used for brightness erosion.",
            ),
            ParameterSpec(
                name="feedback",
                kind=ParameterKind.BOOL,
                default=False,
                label="Feedback",
                description="When enabled, read from the progressively modified image.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)

        if self.x_shift == 0 and self.y_shift == 0:
            return image.copy()

        source = image.data.copy()
        output = image.data.copy()

        if self.feedback:
            source = output

        height, width = image.height, image.width

        x_start = max(0, -self.x_shift)
        x_end = min(width, width - self.x_shift)

        y_start = max(0, -self.y_shift)
        y_end = min(height, height - self.y_shift)

        if x_start >= x_end or y_start >= y_end:
            return image.copy()

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                neighbor_y = y + self.y_shift
                neighbor_x = x + self.x_shift

                current_brightness = np.sum(source[y, x, :3])
                neighbor_brightness = np.sum(source[neighbor_y, neighbor_x, :3])

                if current_brightness > neighbor_brightness:
                    output[y, x] = source[neighbor_y, neighbor_x]

        return image.with_data(output)
