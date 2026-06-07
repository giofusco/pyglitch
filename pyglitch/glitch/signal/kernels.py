from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


NUMBA_AVAILABLE = njit is not None


def _identity_decorator(function):
    return function


_jit = njit(cache=True, fastmath=True) if njit is not None else _identity_decorator


@_jit
def tremolo_kernel(
    signal: np.ndarray,
    frequency: float,
    depth: float,
    sample_rate: float,
) -> np.ndarray:
    source = signal.astype(np.float32)
    output = np.empty_like(source, dtype=np.float32)

    factor = 2.0 * math.pi * (frequency / sample_rate)

    for index in range(source.size):
        modulation = 1.0 + depth * math.sin(index * factor)
        output[index] = source[index] * modulation

    return output


@_jit
def flanger_kernel(
    signal: np.ndarray,
    max_sample_delay: int,
    rate: float,
    sample_rate: float,
    dry: float,
    wet: float,
) -> np.ndarray:
    source = signal.astype(np.float32)
    output = source.copy()

    if max_sample_delay <= 0 or max_sample_delay >= source.size:
        return output

    rate_factor = rate / sample_rate

    for index in range(max_sample_delay, source.size):
        lfo = abs(math.sin(2.0 * math.pi * index * rate_factor))
        current_delay = int(math.ceil(lfo * max_sample_delay))

        if current_delay <= 0:
            delayed_value = source[index]
        else:
            delayed_value = source[index - current_delay]

        output[index] = dry * source[index] + wet * delayed_value

    return output


@_jit
def reverb_kernel(
    signal: np.ndarray,
    delay_pixels: int,
    decay: float,
    feedback: bool,
) -> np.ndarray:
    output = signal.astype(np.float32).copy()
    delay = abs(delay_pixels)

    if delay_pixels == 0:
        return output

    if delay >= output.size:
        return output

    if delay_pixels > 0:
        for index in range(0, output.size - delay):
            if feedback:
                source_value = output[index]
            else:
                source_value = signal[index]

            output[index + delay] += source_value * decay
    else:
        for index in range(output.size - 1, delay - 1, -1):
            if feedback:
                source_value = output[index]
            else:
                source_value = signal[index]

            output[index - delay] += source_value * decay

    return output


@_jit
def wah_wah_kernel(
    signal: np.ndarray,
    center_frequencies: np.ndarray,
    damping: float,
    sample_rate: float,
) -> np.ndarray:
    if signal.size < 3:
        return signal.astype(np.float32).copy()

    source = signal.astype(np.float32)

    high = np.zeros(signal.size, dtype=np.float32)
    band = np.zeros(signal.size, dtype=np.float32)
    low = np.zeros(signal.size, dtype=np.float32)

    high[1] = source[1]

    first_coefficient = 2.0 * math.sin(
        (math.pi * center_frequencies[1]) / sample_rate
    )
    band[1] = first_coefficient * high[1]
    low[1] = first_coefficient * band[1]

    resonance = 2.0 * damping

    for index in range(2, signal.size):
        coefficient = 2.0 * math.sin(
            (math.pi * center_frequencies[index]) / sample_rate
        )

        high[index] = source[index] - low[index - 1] - resonance * band[index - 1]
        band[index] = coefficient * high[index] + band[index - 1]
        low[index] = coefficient * band[index] + low[index - 1]

    max_abs = 0.0
    for index in range(band.size):
        value = abs(band[index])
        if value > max_abs:
            max_abs = value

    if max_abs == 0.0:
        return np.zeros_like(signal, dtype=np.float32)

    output = np.empty_like(signal, dtype=np.float32)

    for index in range(signal.size):
        normalized = band[index] / max_abs
        output[index] = (normalized + 1.0) * 0.5

    return output
