from __future__ import annotations

from abc import ABC, abstractmethod

from pyglitch.glitch.parameters import Parameterized
from pyglitch.image.glitch_image import GlitchImage


class GlitchFilter(ABC, Parameterized):
    """Base class for declared image transformations."""

    @abstractmethod
    def apply(self, image: GlitchImage) -> GlitchImage:
        """Return a transformed copy of the input image."""
        raise NotImplementedError
