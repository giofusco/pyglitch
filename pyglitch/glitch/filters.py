from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.utils import validate_uint8_image
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class ConvolutionFilter(GlitchFilter):
    """Apply a 2D convolution kernel to every image channel."""

    kernel: np.ndarray

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="kernel",
                kind=ParameterKind.ARRAY,
                default=None,
                label="Kernel",
                description="2D convolution kernel.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)

        if self.kernel.ndim != 2:
            raise ValueError(
                f"Expected 2D filter kernel, got shape {self.kernel.shape}."
            )

        source = image.data.astype(np.float32)
        filtered = np.empty_like(source, dtype=np.float32)

        for channel in range(image.num_channels):
            filtered[:, :, channel] = signal.convolve2d(
                source[:, :, channel],
                self.kernel,
                mode="same",
                boundary="symm",
            )

        data = np.clip(filtered, 0, 255).round().astype(np.uint8)

        return image.with_data(data)
