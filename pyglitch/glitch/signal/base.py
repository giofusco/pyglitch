from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.parameters import ParameterKind, ParameterSpec
from pyglitch.glitch.signal.scan import image_to_signal, signal_to_image
from pyglitch.glitch.signal.types import (
    SignalOutputMode,
    SignalRebuildMode,
    SignalScanMode,
)
from pyglitch.glitch.utils import validate_uint8_image
from pyglitch.image.glitch_image import GlitchImage


_UINT8_MAX = 255.0


@dataclass(frozen=True)
class SignalFilterBase(GlitchFilter):
    """Base class for filters that treat image data as a 1D float signal.

    Signal filters operate in normalized float32 image space: [0, 1].  The only
    uint8 boundary is the public ``apply`` input/output boundary.  This keeps
    audio-style signal operations from wrapping, quantizing, or losing negative
    and overshoot values before the final output mapping.

    live_scale and mix are intentionally approximate/live-performance controls:

    - live_scale < 1.0 processes a smaller image, then upscales the result.
    - mix < 1.0 blends the processed result back with the original image.

    These are not full-fidelity image processing options. They are meant for
    live glitch rendering, where responsiveness and motion are often more
    important than exact reconstruction.
    """

    scan_mode: SignalScanMode = SignalScanMode.ROW_MAJOR
    rebuild_mode: SignalRebuildMode = SignalRebuildMode.MATCH_SCAN
    output_mode: SignalOutputMode = SignalOutputMode.CLIP
    live_scale: float = 1.0
    mix: float = 1.0

    @classmethod
    def signal_parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        """Shared signal-routing parameters exposed by all signal filters."""
        return (
            ParameterSpec(
                name="scan_mode",
                kind=ParameterKind.ENUM,
                default=SignalScanMode.ROW_MAJOR,
                enum_values=tuple(SignalScanMode),
                label="Scan Mode",
                description="How image pixels are read into signal form.",
            ),
            ParameterSpec(
                name="rebuild_mode",
                kind=ParameterKind.ENUM,
                default=SignalRebuildMode.MATCH_SCAN,
                enum_values=tuple(SignalRebuildMode),
                label="Rebuild Mode",
                description="How a processed full-image signal is written back.",
            ),
            ParameterSpec(
                name="output_mode",
                kind=ParameterKind.ENUM,
                default=SignalOutputMode.CLIP,
                enum_values=tuple(SignalOutputMode),
                label="Output Mode",
                description="How processed float values are converted back to uint8.",
            ),
            ParameterSpec(
                name="live_scale",
                kind=ParameterKind.FLOAT,
                minimum=0.05,
                maximum=1.0,
                default=1.0,
                label="Live Scale",
                description="Approximate mode: process at a lower scale and upscale.",
            ),
            ParameterSpec(
                name="mix",
                kind=ParameterKind.FLOAT,
                minimum=0.0,
                maximum=1.0,
                default=1.0,
                label="Mix",
                description="Blend between original image and processed result.",
            ),
        )

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return cls.signal_parameter_specs()

    def apply(self, image: GlitchImage) -> GlitchImage:
        validate_uint8_image(image)
        self._validate_live_controls()

        original_uint8 = image.data
        original_float = _uint8_to_float(original_uint8)

        if self.live_scale < 1.0:
            working_float = _resize_by_scale(original_float, self.live_scale, order=0)
        else:
            working_float = original_float

        processed_float = self._process_image_data(working_float)
        mapped_float = self._map_output(processed_float)

        if mapped_float.shape != original_float.shape:
            mapped_float = _resize_to_shape(
                mapped_float,
                original_float.shape,
                order=1,
            )
            mapped_float = np.clip(mapped_float, 0.0, 1.0)

        if self.mix < 1.0:
            mapped_float = _blend_float(
                original=original_float,
                processed=mapped_float,
                mix=self.mix,
            )

        return image.with_data(_float_to_uint8(mapped_float))

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _process_image_data(self, data: np.ndarray) -> np.ndarray:
        float_data = _ensure_float32(data)

        if self.scan_mode is SignalScanMode.ROWS_INDEPENDENT:
            return self._process_rows_independent(float_data)

        if self.scan_mode is SignalScanMode.COLUMNS_INDEPENDENT:
            return self._process_columns_independent(float_data)

        signal = image_to_signal(float_data, self.scan_mode)
        processed_signal = self.process_signal(signal)

        return signal_to_image(
            signal=processed_signal,
            shape=float_data.shape,
            scan_mode=self.scan_mode,
            rebuild_mode=self.rebuild_mode,
        )

    def _process_rows_independent(self, data: np.ndarray) -> np.ndarray:
        _validate_channel_last_image(data)
        output = np.empty_like(data, dtype=np.float32)

        for row_index in range(data.shape[0]):
            row_signal = data[row_index, :, :].reshape(-1).astype(np.float32)
            processed = self.process_signal(row_signal)
            output[row_index, :, :] = processed.reshape(data.shape[1], data.shape[2])

        return output

    def _process_columns_independent(self, data: np.ndarray) -> np.ndarray:
        _validate_channel_last_image(data)
        output = np.empty_like(data, dtype=np.float32)

        for column_index in range(data.shape[1]):
            column_signal = data[:, column_index, :].reshape(-1).astype(np.float32)
            processed = self.process_signal(column_signal)
            output[:, column_index, :] = processed.reshape(data.shape[0], data.shape[2])

        return output

    def _validate_live_controls(self) -> None:
        if not 0.0 < self.live_scale <= 1.0:
            raise ValueError(f"live_scale must be in (0, 1], got {self.live_scale}.")

        if not 0.0 <= self.mix <= 1.0:
            raise ValueError(f"mix must be in [0, 1], got {self.mix}.")

    def _map_output(self, data: np.ndarray) -> np.ndarray:
        """Map processed float signal values to displayable [0, 1] image space."""
        float_data = _ensure_float32(data)

        if self.output_mode is SignalOutputMode.CLIP:
            return np.clip(float_data, 0.0, 1.0)

        if self.output_mode is SignalOutputMode.RESCALE:
            minimum = float(float_data.min())
            maximum = float(float_data.max())

            if minimum == maximum:
                return np.zeros_like(float_data, dtype=np.float32)

            scaled = (float_data - minimum) / (maximum - minimum)
            return np.clip(scaled, 0.0, 1.0).astype(np.float32)

        raise ValueError(f"Unsupported output mode: {self.output_mode}.")


def _uint8_to_float(data: np.ndarray) -> np.ndarray:
    return data.astype(np.float32) / _UINT8_MAX


def _float_to_uint8(data: np.ndarray) -> np.ndarray:
    return np.round(np.clip(data, 0.0, 1.0) * _UINT8_MAX).astype(np.uint8)


def _ensure_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.float32:
        return data

    return data.astype(np.float32)


def _validate_channel_last_image(data: np.ndarray) -> None:
    if data.ndim != 3:
        raise ValueError(
            "Independent row/column signal modes require channel-last image data "
            f"with shape (height, width, channels), got {data.shape}."
        )


def _resize_by_scale(
    data: np.ndarray,
    scale: float,
    order: int,
) -> np.ndarray:
    height = max(1, int(round(data.shape[0] * scale)))
    width = max(1, int(round(data.shape[1] * scale)))

    return _resize_to_shape(data, (height, width, data.shape[2]), order=order)


def _resize_to_shape(
    data: np.ndarray,
    shape: tuple[int, ...],
    order: int,
) -> np.ndarray:
    zoom = (
        shape[0] / float(data.shape[0]),
        shape[1] / float(data.shape[1]),
        1.0,
    )

    resized = ndimage.zoom(data, zoom=zoom, order=order)

    # ndimage can be off by one due to rounding. Crop/pad defensively.
    output = np.zeros(shape, dtype=resized.dtype)

    copy_height = min(shape[0], resized.shape[0])
    copy_width = min(shape[1], resized.shape[1])

    output[:copy_height, :copy_width, :] = resized[:copy_height, :copy_width, :]

    return output.astype(np.float32, copy=False)


def _blend_float(
    original: np.ndarray,
    processed: np.ndarray,
    mix: float,
) -> np.ndarray:
    blended = (1.0 - mix) * original + mix * processed
    return np.clip(blended, 0.0, 1.0).astype(np.float32)
