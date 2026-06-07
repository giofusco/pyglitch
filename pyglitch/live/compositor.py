from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from pyglitch.image.glitch_image import GlitchImage


class FrameBlendMode(str, Enum):
    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"


@dataclass
class FrameCompositor:
    mode: FrameBlendMode = FrameBlendMode.CROSSFADE
    half_life: float = 0.08
    current: GlitchImage | None = None

    def reset(self, image: GlitchImage) -> None:
        self.current = image.copy()

    def compose(self, target: GlitchImage, dt: float) -> GlitchImage:
        if self.current is None or self.current.shape != target.shape:
            self.current = target.copy()
            return self.current
        if self.mode is FrameBlendMode.HARD_CUT:
            self.current = target.copy()
            return self.current
        if self.mode is FrameBlendMode.CROSSFADE:
            alpha = self._alpha_from_dt(dt)
            self.current = self.current.with_data(blend_arrays(self.current.data, target.data, alpha))
            return self.current
        raise ValueError(f"Unsupported frame blend mode: {self.mode}")

    def _alpha_from_dt(self, dt: float) -> float:
        if self.half_life <= 0:
            return 1.0
        return 1.0 - math.exp(-0.69314718056 * max(0.0, dt) / self.half_life)


def blend_arrays(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(1.0, max(0.0, float(alpha)))
    data = (1.0 - alpha) * first.astype(np.float32) + alpha * second.astype(np.float32)
    return np.clip(data, 0, 255).round().astype(np.uint8)
