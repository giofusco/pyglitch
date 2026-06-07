# Copyright 2017-2026 Giovanni Fusco
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg

from pyglitch.image.ops import to_uint8_image
from pyglitch.image.types import ImageArray


def load_image(filename: str | Path) -> ImageArray:
    """Load image as a uint8 NumPy array."""

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = mpimg.imread(path)

    return to_uint8_image(image)


def save_image(image: ImageArray, filename: str | Path) -> None:
    """Save image to disk."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    mpimg.imsave(path, image)