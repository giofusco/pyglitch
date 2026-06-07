from __future__ import annotations

from dataclasses import dataclass

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.image.glitch_image import GlitchImage
from pyglitch.image.types import Patch, Rect


@dataclass(frozen=True)
class ApplyToRect(GlitchFilter):
    """Apply another filter to a rectangular patch."""

    rect: Rect
    filter: GlitchFilter

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="rect",
                kind=ParameterKind.RECT,
                default=None,
                label="Rectangle",
                description="Image region where the nested filter is applied.",
            ),
            ParameterSpec(
                name="filter",
                kind=ParameterKind.FILTER,
                default=None,
                label="Filter",
                description="Nested filter applied to the rectangle.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        result = image.copy()
        patch = result.get_patch(self.rect)

        patch_image = GlitchImage(
            data=patch.data.copy(),
            name=image.name,
            source_path=image.source_path,
            frame_index=image.frame_index,
            metadata=image.metadata.copy(),
        )

        transformed_patch_image = self.filter.apply(patch_image)

        result.put_patch(
            Patch(
                rect=patch.rect,
                data=transformed_patch_image.data,
            )
        )

        return result
