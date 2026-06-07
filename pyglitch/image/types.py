# Copyright 2017-2026 Giovanni Fusco
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangular region in image coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class Patch:
    """Image patch with its original location."""

    rect: Rect
    data: ImageArray
