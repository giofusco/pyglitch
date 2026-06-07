from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.signal.base import SignalFilterBase
from pyglitch.glitch.signal.kernels import tremolo_kernel


@dataclass(frozen=True)
class SignalTremolo(SignalFilterBase):
    """Amplitude-modulate the flattened image signal."""

    frequency: float = 5.0
    depth: float = 0.5
    sample_rate: float = 44_100.0

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="frequency",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=200.0,
                default=5.0,
                label="Frequency",
                description="Tremolo modulation frequency.",
            ),
            ParameterSpec(
                name="depth",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=0.5,
                label="Depth",
                description="Tremolo modulation depth.",
            ),
            ParameterSpec(
                name="sample_rate",
                kind=ParameterKind.FLOAT,
                minimum=1.0,
                maximum=192000.0,
                default=44100.0,
                label="Sample Rate",
                description="Virtual sample rate used by the signal tremolo.",
            ),
            *cls.signal_parameter_specs(),
        )

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}.")

        if not 0.0 <= self.depth <= 1.0:
            raise ValueError(f"depth must be in [0, 1], got {self.depth}.")

        return tremolo_kernel(
            signal=signal,
            frequency=float(self.frequency),
            depth=float(self.depth),
            sample_rate=float(self.sample_rate),
        )
