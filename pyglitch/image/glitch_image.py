# Copyright 2017-2026 Giovanni Fusco
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from numpy import ndarray

from pyglitch.image.io import load_image, save_image
from pyglitch.image.ops import (
    flatten_image,
    get_patch,
    image_height,
    image_width,
    num_channels,
    put_patch_in_place,
    rotate_left,
    rotate_right,
    swap_patches_in_place,
)
from pyglitch.image.types import ImageArray, Patch, Rect


@dataclass
class GlitchImage:
    """First-class image object used by pyglitch."""

    data: ImageArray
    name: str | None = None
    source_path: Path | None = None
    frame_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, filename: str | Path) -> "GlitchImage":
        path = Path(filename)

        return cls(
            data=load_image(path),
            name=path.stem,
            source_path=path,
        )

    def save(self, filename: str | Path) -> None:
        save_image(self.data, filename)

    @property
    def width(self) -> int:
        return image_width(self.data)

    @property
    def height(self) -> int:
        return image_height(self.data)

    @property
    def num_channels(self) -> int:
        return num_channels(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def copy(self) -> "GlitchImage":
        return GlitchImage(
            data=self.data.copy(),
            name=self.name,
            source_path=self.source_path,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )

    def rotated_left(self) -> "GlitchImage":
        return GlitchImage(
            data=rotate_left(self.data),
            name=self.name,
            source_path=self.source_path,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )

    def rotated_right(self) -> "GlitchImage":
        return GlitchImage(
            data=rotate_right(self.data),
            name=self.name,
            source_path=self.source_path,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )

    def rotate_left_in_place(self) -> None:
        self.data = rotate_left(self.data)

    def rotate_right_in_place(self) -> None:
        self.data = rotate_right(self.data)

    def get_patch(self, rect: Rect) -> Patch:
        return get_patch(self.data, rect)

    def put_patch(self, patch: Patch) -> None:
        put_patch_in_place(self.data, patch)

    def swap_patches(self, first_patch: Patch, second_patch: Patch) -> None:
        swap_patches_in_place(self.data, first_patch, second_patch)

    def flatten(self) -> NDArray:
        return flatten_image(self.data)
    
    def with_data(self, data: ImageArray) -> "GlitchImage":
        return GlitchImage(
            data=data,
            name=self.name,
            source_path=self.source_path,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )
