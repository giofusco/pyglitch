from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.image.glitch_image import GlitchImage


class TemporalRevealMode(str, Enum):
    CROSSFADE = "crossfade"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"
    CENTER_OUT = "center_out"
    RANDOM_BLOCKS = "random_blocks"


@dataclass(frozen=True)
class BlendToFilter(GlitchFilter):
    """Blend between the original image and a filtered target.

    progress=0 returns the original image.
    progress=1 returns the fully filtered image.
    """

    filter: GlitchFilter
    progress: float = 1.0

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="filter",
                kind=ParameterKind.FILTER,
                default=None,
                label="Filter",
                description="Filter used to produce the target image.",
            ),
            ParameterSpec(
                name="progress",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=1.0,
                label="Progress",
                description="Blend progress between original and filtered image.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        progress = _clamp01(self.progress)

        if progress == 0.0:
            return image.copy()

        target = self.filter.apply(image)

        if progress == 1.0:
            return target

        data = blend_images(image.data, target.data, progress)

        return image.with_data(data)


@dataclass(frozen=True)
class ProgressiveRevealFilter(GlitchFilter):
    """Reveal a filtered target progressively using a spatial mask.

    This is a stateless filter wrapper. For live performance, prefer
    FilterTransitionRenderer, which computes the target once and then animates
    cached frames.
    """

    filter: GlitchFilter
    progress: float = 1.0
    mode: TemporalRevealMode = TemporalRevealMode.LEFT_TO_RIGHT
    block_size: int = 32
    random_seed: int = 0

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="filter",
                kind=ParameterKind.FILTER,
                default=None,
                label="Filter",
                description="Filter used to produce the target image.",
            ),
            ParameterSpec(
                name="progress",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=1.0,
                label="Progress",
                description="Reveal progress from original to target.",
            ),
            ParameterSpec(
                name="mode",
                kind=ParameterKind.ENUM,
                default=TemporalRevealMode.LEFT_TO_RIGHT,
                enum_values=tuple(TemporalRevealMode),
                label="Reveal Mode",
                description="Spatial reveal pattern.",
            ),
            ParameterSpec(
                name="block_size",
                kind=ParameterKind.INT,
                minimum=1,
                maximum=512,
                default=32,
                label="Block Size",
                description="Block size used by random block reveal.",
            ),
            ParameterSpec(
                name="random_seed",
                kind=ParameterKind.INT,
                minimum=0,
                maximum=2**31 - 1,
                default=0,
                label="Random Seed",
                description="Seed used by random block reveal.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        progress = _clamp01(self.progress)

        if progress == 0.0:
            return image.copy()

        target = self.filter.apply(image)

        if progress == 1.0:
            return target

        data = reveal_images(
            original=image.data,
            target=target.data,
            progress=progress,
            mode=self.mode,
            block_size=self.block_size,
            random_seed=self.random_seed,
        )

        return image.with_data(data)


def blend_images(
    original: np.ndarray,
    target: np.ndarray,
    progress: float,
) -> np.ndarray:
    progress = _clamp01(progress)

    data = (
        (1.0 - progress) * original.astype(np.float32)
        + progress * target.astype(np.float32)
    )

    return np.clip(data, 0, 255).round().astype(np.uint8)


def reveal_images(
    original: np.ndarray,
    target: np.ndarray,
    progress: float,
    mode: TemporalRevealMode,
    block_size: int = 32,
    random_seed: int = 0,
) -> np.ndarray:
    progress = _clamp01(progress)

    if progress <= 0.0:
        return original.copy()

    if progress >= 1.0:
        return target.copy()

    mask = reveal_mask(
        shape=original.shape[:2],
        progress=progress,
        mode=mode,
        block_size=block_size,
        random_seed=random_seed,
    )

    output = original.copy()
    output[mask, :] = target[mask, :]

    return output


def reveal_mask(
    shape: tuple[int, int],
    progress: float,
    mode: TemporalRevealMode,
    block_size: int = 32,
    random_seed: int = 0,
) -> np.ndarray:
    height, width = shape
    progress = _clamp01(progress)

    if mode is TemporalRevealMode.CROSSFADE:
        # For mask users, crossfade behaves like a full reveal at progress 1 only.
        return np.full((height, width), progress >= 1.0, dtype=bool)

    if mode is TemporalRevealMode.LEFT_TO_RIGHT:
        limit = int(round(width * progress))
        mask = np.zeros((height, width), dtype=bool)
        mask[:, :limit] = True
        return mask

    if mode is TemporalRevealMode.RIGHT_TO_LEFT:
        limit = int(round(width * progress))
        mask = np.zeros((height, width), dtype=bool)
        if limit > 0:
            mask[:, width - limit :] = True
        return mask

    if mode is TemporalRevealMode.TOP_TO_BOTTOM:
        limit = int(round(height * progress))
        mask = np.zeros((height, width), dtype=bool)
        mask[:limit, :] = True
        return mask

    if mode is TemporalRevealMode.BOTTOM_TO_TOP:
        limit = int(round(height * progress))
        mask = np.zeros((height, width), dtype=bool)
        if limit > 0:
            mask[height - limit :, :] = True
        return mask

    if mode is TemporalRevealMode.CENTER_OUT:
        y_positions, x_positions = np.ogrid[:height, :width]
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0

        distances = np.sqrt(
            (y_positions - center_y) ** 2
            + (x_positions - center_x) ** 2
        )

        max_distance = float(distances.max())

        if max_distance == 0.0:
            return np.ones((height, width), dtype=bool)

        return distances <= max_distance * progress

    if mode is TemporalRevealMode.RANDOM_BLOCKS:
        return _random_block_mask(
            shape=shape,
            progress=progress,
            block_size=block_size,
            random_seed=random_seed,
        )

    raise ValueError(f"Unsupported reveal mode: {mode}.")


def _random_block_mask(
    shape: tuple[int, int],
    progress: float,
    block_size: int,
    random_seed: int,
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    height, width = shape

    block_rows = int(np.ceil(height / block_size))
    block_cols = int(np.ceil(width / block_size))
    total_blocks = block_rows * block_cols
    reveal_count = int(round(total_blocks * progress))

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(total_blocks)
    selected = set(int(index) for index in order[:reveal_count])

    mask = np.zeros((height, width), dtype=bool)

    for block_index in selected:
        block_row = block_index // block_cols
        block_col = block_index % block_cols

        y0 = block_row * block_size
        x0 = block_col * block_size
        y1 = min(y0 + block_size, height)
        x1 = min(x0 + block_size, width)

        mask[y0:y1, x0:x1] = True

    return mask


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
