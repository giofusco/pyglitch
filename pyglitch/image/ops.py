# Copyright 2017-2026 Giovanni Fusco
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pyglitch.image.types import ImageArray, Patch, Rect


ColorChannel = Literal[0, 1, 2]

CH_RED: ColorChannel = 0
CH_GREEN: ColorChannel = 1
CH_BLUE: ColorChannel = 2


def image_width(image: NDArray) -> int:
    return int(image.shape[1])


def image_height(image: NDArray) -> int:
    return int(image.shape[0])


def num_channels(image: NDArray) -> int:
    if image.ndim < 3:
        return 1

    return int(image.shape[2])


def rotate_left(image: ImageArray) -> ImageArray:
    """Return image rotated 90 degrees counter-clockwise."""

    return np.rot90(image, k=1).copy()


def rotate_right(image: ImageArray) -> ImageArray:
    """Return image rotated 90 degrees clockwise."""

    return np.rot90(image, k=-1).copy()


def is_rect_inside_image(image: NDArray, rect: Rect) -> bool:
    """Return True when rect is fully contained in image."""

    if rect.x < 0 or rect.y < 0:
        return False

    if rect.width <= 0 or rect.height <= 0:
        return False

    return rect.x_end <= image_width(image) and rect.y_end <= image_height(image)


def validate_rect_inside_image(image: NDArray, rect: Rect) -> None:
    """Raise ValueError if rect is outside image bounds."""

    if is_rect_inside_image(image, rect):
        return

    raise ValueError(
        "Rectangle is outside image bounds: "
        f"rect={rect}, image_size=({image_width(image)}, {image_height(image)})"
    )


def extract_block(image: ImageArray, rect: Rect) -> ImageArray:
    """Extract a copied rectangular image block."""

    validate_rect_inside_image(image, rect)

    return image[
        rect.y : rect.y_end,
        rect.x : rect.x_end,
    ].copy()


def get_patch(image: ImageArray, rect: Rect) -> Patch:
    """Extract a patch and remember its original location."""

    return Patch(
        rect=rect,
        data=extract_block(image, rect),
    )


def put_patch_in_place(image: ImageArray, patch: Patch) -> None:
    """Overwrite image content with patch data at the patch location."""

    validate_rect_inside_image(image, patch.rect)
    validate_patch_shape(image, patch)

    image[
        patch.rect.y : patch.rect.y_end,
        patch.rect.x : patch.rect.x_end,
    ] = patch.data


def swap_patches_in_place(
    image: ImageArray,
    first_patch: Patch,
    second_patch: Patch,
) -> None:
    """Swap two same-sized patches inside image."""

    validate_rect_inside_image(image, first_patch.rect)
    validate_rect_inside_image(image, second_patch.rect)
    validate_patch_shape(image, first_patch)
    validate_patch_shape(image, second_patch)

    if first_patch.data.shape != second_patch.data.shape:
        raise ValueError(
            "Cannot swap patches with different shapes: "
            f"{first_patch.data.shape} != {second_patch.data.shape}"
        )

    image[
        first_patch.rect.y : first_patch.rect.y_end,
        first_patch.rect.x : first_patch.rect.x_end,
    ] = second_patch.data

    image[
        second_patch.rect.y : second_patch.rect.y_end,
        second_patch.rect.x : second_patch.rect.x_end,
    ] = first_patch.data


def swapped_patches(
    image: ImageArray,
    first_patch: Patch,
    second_patch: Patch,
) -> ImageArray:
    """Return a copied image with two patches swapped."""

    output = image.copy()
    swap_patches_in_place(output, first_patch, second_patch)

    return output


def flatten_image(image: NDArray) -> NDArray:
    """Return image as a contiguous 1D array."""

    return np.asarray(image).ravel()


def validate_patch_shape(image: ImageArray, patch: Patch) -> None:
    target_shape = image[
        patch.rect.y : patch.rect.y_end,
        patch.rect.x : patch.rect.x_end,
    ].shape

    if patch.data.shape == target_shape:
        return

    raise ValueError(
        f"Patch data shape {patch.data.shape} does not match target shape {target_shape}"
    )


def to_uint8_image(image: NDArray) -> ImageArray:
    """Convert common image array formats to uint8 safely."""

    if image.dtype == np.uint8:
        return image.copy()

    if np.issubdtype(image.dtype, np.floating):
        clipped = np.clip(image, 0.0, 1.0)
        return (clipped * 255.0).round().astype(np.uint8)

    clipped = np.clip(image, 0, 255)

    return clipped.astype(np.uint8)