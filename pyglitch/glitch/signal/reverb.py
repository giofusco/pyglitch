from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.signal.base import SignalFilterBase
from pyglitch.glitch.signal.kernels import reverb_kernel


@dataclass(frozen=True)
class SignalReverb(SignalFilterBase):
    """Add delayed signal energy along the chosen image scan path."""

    delay_pixels: int = 128
    decay: float = 0.5
    feedback: bool = True

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="delay_pixels",
                kind=ParameterKind.INT,
                minimum=-20000,
                maximum=20000,
                default=128,
                label="Delay Pixels",
                description="Delay offset in flattened signal samples. Must be non-zero.",
            ),
            ParameterSpec(
                name="decay",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=0.5,
                label="Decay",
                description="Amount of delayed signal added back into the image signal.",
            ),
            ParameterSpec(
                name="feedback",
                kind=ParameterKind.BOOL,
                default=True,
                label="Feedback",
                description="Use processed signal as the delay source for recursive feedback.",
            ),
            *cls.signal_parameter_specs(),
        )

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        if self.delay_pixels == 0:
            raise ValueError("delay_pixels must be non-zero.")

        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {self.decay}.")

        return reverb_kernel(
            signal=signal,
            delay_pixels=int(self.delay_pixels),
            decay=float(self.decay),
            feedback=bool(self.feedback),
        )
