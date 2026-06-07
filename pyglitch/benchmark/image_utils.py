from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from pyglitch.image import GlitchImage


def make_preview_images(
    image: GlitchImage,
    widths: tuple[int, ...],
    include_full: bool = False,
) -> dict[str, GlitchImage]:
    previews: dict[str, GlitchImage] = {}

    for width in widths:
        previews[f"w{width}"] = resize_to_width(image, width)

    if include_full:
        previews["full"] = image.copy()

    return previews


def resize_to_width(
    image: GlitchImage,
    width: int,
) -> GlitchImage:
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")

    if image.width == width:
        return image.copy()

    scale = width / float(image.width)
    height = max(1, int(round(image.height * scale)))

    zoom = (
        height / float(image.height),
        width / float(image.width),
        1.0,
    )

    resized = ndimage.zoom(
        image.data,
        zoom=zoom,
        order=1,
    )

    resized = np.clip(resized, 0, 255).round().astype(np.uint8)

    return image.with_data(resized)


def timestamped_report_path(
    directory: str | Path = "reports",
    prefix: str = "filter_benchmark",
    suffix: str = ".csv",
) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(directory) / f"{prefix}_{timestamp}{suffix}"
