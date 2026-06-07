from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.signal.base import SignalFilterBase
from pyglitch.glitch.signal.kernels import flanger_kernel


@dataclass(frozen=True)
class SignalFlanger(SignalFilterBase):
    """Mix the signal with a short time-varying delayed copy."""

    max_time_delay: float = 0.003
    rate: float = 1.0
    sample_rate: float = 44_100.0
    dry: float = 1.0
    wet: float = 0.7

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="max_time_delay",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=0.05,
                default=0.003,
                label="Max Time Delay",
                description="Maximum delay time in seconds.",
            ),
            ParameterSpec(
                name="rate",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=50.0,
                default=1.0,
                label="Rate",
                description="Delay modulation rate.",
            ),
            ParameterSpec(
                name="sample_rate",
                kind=ParameterKind.FLOAT,
                minimum=1.0,
                maximum=192000.0,
                default=44100.0,
                label="Sample Rate",
                description="Virtual sample rate used to convert time delay to samples.",
            ),
            ParameterSpec(
                name="dry",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=2.0,
                default=1.0,
                label="Dry",
                description="Amount of original signal preserved.",
            ),
            ParameterSpec(
                name="wet",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=2.0,
                default=0.7,
                label="Wet",
                description="Amount of delayed signal mixed in.",
            ),
            *cls.signal_parameter_specs(),
        )

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        if self.max_time_delay < 0:
            raise ValueError(
                f"max_time_delay must be non-negative, got {self.max_time_delay}."
            )

        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}.")

        max_sample_delay = int(round(self.max_time_delay * self.sample_rate))

        return flanger_kernel(
            signal=signal,
            max_sample_delay=max_sample_delay,
            rate=float(self.rate),
            sample_rate=float(self.sample_rate),
            dry=float(self.dry),
            wet=float(self.wet),
        )
