from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.utils import validate_image, validate_uint8_image
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class RescaleImage(GlitchFilter):
    """Rescale image values to uint8 [0, 255] when they exceed that range."""

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return ()

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_image(image)

        data = image.data.astype(np.float32)

        minimum = data.min()
        maximum = data.max()

        if minimum == maximum:
            return image.with_data(np.zeros_like(image.data, dtype=np.uint8))

        if minimum < 0 or maximum > 255:
            data = (data - minimum) * (255.0 / (maximum - minimum))

        return image.with_data(np.clip(data, 0, 255).round().astype(np.uint8))


@dataclass(frozen=True)
class Posterize(GlitchFilter):
    """Quantize image values to a fixed number of bins per channel."""

    bins: int
    normalize: bool = False

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="bins",
                kind=ParameterKind.INT,
                minimum=2,
                maximum=64,
                default=8,
                label="Bins",
                description="Number of quantization levels per channel.",
            ),
            ParameterSpec(
                name="normalize",
                kind=ParameterKind.BOOL,
                default=False,
                label="Normalize",
                description="Rescale image values before quantization.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)

        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}.")

        source = RescaleImage().apply(image).data if self.normalize else image.data

        normalized = source.astype(np.float32) / 255.0
        quantized = np.round(normalized * (self.bins - 1)) / (self.bins - 1)
        data = np.clip(quantized * 255.0, 0, 255).astype(np.uint8)

        return image.with_data(data)
