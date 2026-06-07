from __future__ import annotations

import numpy as np

from pyglitch.image.glitch_image import GlitchImage


def validate_image(image: GlitchImage) -> None:
    if image.data.ndim != 3:
        raise ValueError(
            f"Expected image shape (height, width, channels), got {image.shape}."
        )

    if image.num_channels < 3:
        raise ValueError(f"Expected at least 3 channels, got {image.num_channels}.")


def validate_uint8_image(image: GlitchImage) -> None:
    validate_image(image)

    if image.data.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.data.dtype}.")


def validate_channel(image: GlitchImage, channel: int) -> None:
    if not 0 <= channel < image.num_channels:
        raise ValueError(
            f"Invalid channel index {channel}. "
            f"Image has {image.num_channels} channels."
        )


def validate_8bit_value(value: int, name: str) -> None:
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be in [0, 255], got {value}.")
