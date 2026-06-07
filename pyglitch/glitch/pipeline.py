from __future__ import annotations

from dataclasses import dataclass

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.image.glitch_image import GlitchImage


@dataclass(frozen=True)
class GlitchPipeline(GlitchFilter):
    """Ordered list of filters applied one after another."""

    filters: tuple[GlitchFilter, ...]

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="filters",
                kind=ParameterKind.FILTER_TUPLE,
                default=(),
                label="Filters",
                description="Ordered filters applied by this pipeline.",
            ),
        )

    def apply(self, image: GlitchImage) -> GlitchImage:
        result = image

        for filter_ in self.filters:
            result = filter_.apply(result)

        return result
