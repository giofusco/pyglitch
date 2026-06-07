from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.types import PixelSortKey
from pyglitch.glitch.utils import validate_8bit_value, validate_uint8_image
from pyglitch.image.glitch_image import GlitchImage


def _pixel_sort_parameter_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(
            name="red_threshold",
            kind=ParameterKind.INT,
            minimum=0,
            maximum=255,
            default=128,
            label="Red Threshold",
            description="Red channel brightness threshold.",
        ),
        ParameterSpec(
            name="green_threshold",
            kind=ParameterKind.INT,
            minimum=0,
            maximum=255,
            default=128,
            label="Green Threshold",
            description="Green channel brightness threshold.",
        ),
        ParameterSpec(
            name="blue_threshold",
            kind=ParameterKind.INT,
            minimum=0,
            maximum=255,
            default=128,
            label="Blue Threshold",
            description="Blue channel brightness threshold.",
        ),
        ParameterSpec(
            name="strict",
            kind=ParameterKind.BOOL,
            default=False,
            label="Strict",
            description="Use AND thresholding instead of OR thresholding.",
        ),
        ParameterSpec(
            name="key",
            kind=ParameterKind.ENUM,
            default=PixelSortKey.LUMINANCE,
            enum_values=tuple(PixelSortKey),
            label="Sort Key",
            description="Color-derived key used for sorting each bright segment.",
        ),
        ParameterSpec(
            name="reverse",
            kind=ParameterKind.BOOL,
            default=False,
            label="Reverse",
            description="Reverse the segment sort order.",
        ),
    )


@dataclass(frozen=True)
class PixelSortBrightSegments(GlitchFilter):
    """Sort bright contiguous row segments according to a color-derived key."""

    red_threshold: int
    green_threshold: int
    blue_threshold: int
    strict: bool = False
    key: PixelSortKey = PixelSortKey.LUMINANCE
    reverse: bool = False

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return _pixel_sort_parameter_specs()

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)
        validate_8bit_value(self.red_threshold, "red_threshold")
        validate_8bit_value(self.green_threshold, "green_threshold")
        validate_8bit_value(self.blue_threshold, "blue_threshold")

        data = image.data.copy()

        for row_index in range(image.height):
            row = image.data[row_index, :, :]
            data[row_index, :, :] = _sort_bright_row_segments(
                row=row,
                red_threshold=self.red_threshold,
                green_threshold=self.green_threshold,
                blue_threshold=self.blue_threshold,
                strict=self.strict,
                key=self.key,
                reverse=self.reverse,
            )

        return image.with_data(data)


@dataclass(frozen=True)
class PixelSortBrightSegmentsVertical(GlitchFilter):
    """Sort bright contiguous column segments by rotating the image."""

    red_threshold: int
    green_threshold: int
    blue_threshold: int
    strict: bool = False
    key: PixelSortKey = PixelSortKey.LUMINANCE
    reverse: bool = False

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return _pixel_sort_parameter_specs()

    def apply(self, image: GlitchImage) -> GlitchImage:
        rotated = image.rotated_right()

        sorted_rotated = PixelSortBrightSegments(
            red_threshold=self.red_threshold,
            green_threshold=self.green_threshold,
            blue_threshold=self.blue_threshold,
            strict=self.strict,
            key=self.key,
            reverse=self.reverse,
        ).apply(rotated)

        return sorted_rotated.rotated_left()


def _sort_bright_row_segments(
    row: np.ndarray,
    red_threshold: int,
    green_threshold: int,
    blue_threshold: int,
    strict: bool,
    key: PixelSortKey,
    reverse: bool,
) -> np.ndarray:
    result = row.copy()

    bright_mask = _compute_bright_mask(
        row=row,
        red_threshold=red_threshold,
        green_threshold=green_threshold,
        blue_threshold=blue_threshold,
        strict=strict,
    )

    segment_start: int | None = None

    for index, is_bright in enumerate(bright_mask):
        if is_bright and segment_start is None:
            segment_start = index

        if not is_bright and segment_start is not None:
            _sort_segment_in_place(result, segment_start, index, key, reverse)
            segment_start = None

    if segment_start is not None:
        _sort_segment_in_place(result, segment_start, len(row), key, reverse)

    return result


def _compute_bright_mask(
    row: np.ndarray,
    red_threshold: int,
    green_threshold: int,
    blue_threshold: int,
    strict: bool,
) -> np.ndarray:
    red = row[:, 0] > red_threshold
    green = row[:, 1] > green_threshold
    blue = row[:, 2] > blue_threshold

    if strict:
        return red & green & blue

    return red | green | blue


def _sort_segment_in_place(
    row: np.ndarray,
    start: int,
    end: int,
    key: PixelSortKey,
    reverse: bool,
) -> None:
    if end - start <= 1:
        return

    segment = row[start:end, :]
    sort_values = _compute_sort_values(segment, key)
    indices = np.argsort(sort_values)

    if reverse:
        indices = indices[::-1]

    row[start:end, :] = segment[indices, :]


def _compute_sort_values(
    segment: np.ndarray,
    key: PixelSortKey,
) -> np.ndarray:
    rgb = segment[:, :3].astype(np.float32) / 255.0

    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    if key is PixelSortKey.LUMINANCE:
        return np.sqrt(
            0.241 * red * red
            + 0.691 * green * green
            + 0.068 * blue * blue
        )

    if key is PixelSortKey.VALUE:
        return np.max(rgb, axis=1)

    if key is PixelSortKey.HUE:
        return _compute_hue(red, green, blue)

    raise ValueError(f"Unsupported pixel sort key: {key}.")


def _compute_hue(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
) -> np.ndarray:
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)

    red_is_max = (maximum == red) & (delta != 0)
    green_is_max = (maximum == green) & (delta != 0)
    blue_is_max = (maximum == blue) & (delta != 0)

    hue[red_is_max] = (
        (green[red_is_max] - blue[red_is_max]) / delta[red_is_max]
    ) % 6.0

    hue[green_is_max] = (
        (blue[green_is_max] - red[green_is_max]) / delta[green_is_max]
    ) + 2.0

    hue[blue_is_max] = (
        (red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max]
    ) + 4.0

    return hue / 6.0
