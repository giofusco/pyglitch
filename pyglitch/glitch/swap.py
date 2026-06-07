from __future__ import annotations

from dataclasses import dataclass

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.image.glitch_image import GlitchImage
from pyglitch.image.ops import swapped_patches
from pyglitch.image.types import Rect


@dataclass(frozen=True)
class SwapRects(GlitchFilter):
    first_rect: Rect
    second_rect: Rect

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="first_rect",
                kind=ParameterKind.RECT,
                default=None,
                label="First Rectangle",
                description="First image region to swap.",
            ),
            ParameterSpec(
                name="second_rect",
                kind=ParameterKind.RECT,
                default=None,
                label="Second Rectangle",
                description="Second image region to swap.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        first_patch = image.get_patch(self.first_rect)
        second_patch = image.get_patch(self.second_rect)

        data = swapped_patches(
            image=image.data,
            first_patch=first_patch,
            second_patch=second_patch,
        )

        return image.with_data(data)
