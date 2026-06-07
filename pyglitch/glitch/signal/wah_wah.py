from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.signal.base import SignalFilterBase
from pyglitch.glitch.signal.kernels import wah_wah_kernel


@dataclass(frozen=True)
class SignalWahWah(SignalFilterBase):
    """Apply a swept resonant band-pass filter to the flattened image signal."""

    damping: float = 0.05
    min_frequency: float = 500.0
    max_frequency: float = 5_000.0
    sweep_frequency: float = 2_000.0
    sample_rate: float = 44_100.0

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="damping",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=2.0,
                default=0.05,
                label="Damping",
                description="Band-pass damping factor. Higher values are usually less razor-thin.",
            ),
            ParameterSpec(
                name="min_frequency",
                kind=ParameterKind.FLOAT,
                minimum=1.0,
                maximum=20000.0,
                default=500.0,
                label="Min Frequency",
                description="Minimum swept center frequency.",
            ),
            ParameterSpec(
                name="max_frequency",
                kind=ParameterKind.FLOAT,
                minimum=1.0,
                maximum=20000.0,
                default=5000.0,
                label="Max Frequency",
                description="Maximum swept center frequency.",
            ),
            ParameterSpec(
                name="sweep_frequency",
                kind=ParameterKind.FLOAT,
                minimum=0.01,
                maximum=10000.0,
                default=2000.0,
                label="Sweep Frequency",
                description="Speed of the center-frequency sweep.",
            ),
            ParameterSpec(
                name="sample_rate",
                kind=ParameterKind.FLOAT,
                minimum=1.0,
                maximum=192000.0,
                default=44100.0,
                label="Sample Rate",
                description="Virtual sample rate used by the signal filter.",
            ),
            *cls.signal_parameter_specs(),
        )

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        if signal.size < 3:
            return signal.astype(np.float32).copy()

        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}.")

        if self.min_frequency <= 0 or self.max_frequency <= 0:
            raise ValueError("min_frequency and max_frequency must be positive.")

        if self.max_frequency <= self.min_frequency:
            raise ValueError(
                "max_frequency must be greater than min_frequency: "
                f"{self.max_frequency} <= {self.min_frequency}."
            )

        nyquist_frequency = self.sample_rate * 0.5
        if self.max_frequency >= nyquist_frequency:
            raise ValueError(
                "max_frequency must stay below the Nyquist frequency: "
                f"{self.max_frequency} >= {nyquist_frequency}."
            )

        if self.sweep_frequency <= 0:
            raise ValueError(
                f"sweep_frequency must be positive, got {self.sweep_frequency}."
            )

        if self.damping < 0:
            raise ValueError(f"damping must be non-negative, got {self.damping}.")

        center_frequencies = self._center_frequencies(signal.size)

        return wah_wah_kernel(
            signal=signal,
            center_frequencies=center_frequencies,
            damping=float(self.damping),
            sample_rate=float(self.sample_rate),
        )

    def _center_frequencies(self, signal_size: int) -> np.ndarray:
        delta = self.sweep_frequency / self.sample_rate

        if delta <= 0:
            raise ValueError(f"Invalid frequency sweep delta: {delta}.")

        upward = np.arange(self.min_frequency, self.max_frequency, delta)
        downward = np.arange(self.max_frequency, self.min_frequency, -delta)

        if upward.size == 0 or downward.size == 0:
            raise ValueError("Frequency sweep configuration produced no samples.")

        sweep = np.concatenate([upward, downward])

        repeats = int(math.ceil(signal_size / sweep.size))
        tiled = np.tile(sweep, repeats)

        return tiled[:signal_size].astype(np.float32)
