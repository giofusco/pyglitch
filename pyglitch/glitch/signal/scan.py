from __future__ import annotations

from functools import lru_cache

import numpy as np

from pyglitch.glitch.signal.types import SignalRebuildMode, SignalScanMode


def image_to_signal(
    data: np.ndarray,
    scan_mode: SignalScanMode,
) -> np.ndarray:
    """Read image data into a 1D float32 signal."""
    flat_data = data.reshape(-1)
    indices = scan_indices(data.shape, scan_mode)

    return flat_data[indices].astype(np.float32)


def signal_to_image(
    signal: np.ndarray,
    shape: tuple[int, ...],
    scan_mode: SignalScanMode,
    rebuild_mode: SignalRebuildMode,
) -> np.ndarray:
    """Write a 1D signal back into an image-shaped array."""
    expected_size = int(np.prod(shape))

    if signal.size != expected_size:
        raise ValueError(
            f"Signal size {signal.size} does not match image size {expected_size}."
        )

    output_flat = np.empty(expected_size, dtype=np.float32)

    target_mode = _resolve_rebuild_mode(scan_mode, rebuild_mode)
    target_indices = scan_indices(shape, target_mode)

    output_flat[target_indices] = signal

    return output_flat.reshape(shape)


@lru_cache(maxsize=128)
def scan_indices(
    shape: tuple[int, ...],
    scan_mode: SignalScanMode | SignalRebuildMode,
) -> np.ndarray:
    """Return cached flat indices for a scan/rebuild mode.

    Scan indices depend only on image shape and scan mode, so live rendering can
    reuse them across frames at the same preview resolution.
    """
    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3:
        height, width, channels = shape
    else:
        raise ValueError(f"Expected 2D or 3D image shape, got {shape}.")

    if scan_mode in (SignalScanMode.ROW_MAJOR, SignalRebuildMode.ROW_MAJOR):
        return _row_major_indices(height, width, channels)

    if scan_mode in (SignalScanMode.COLUMN_MAJOR, SignalRebuildMode.COLUMN_MAJOR):
        return _column_major_indices(height, width, channels)

    if scan_mode in (SignalScanMode.SNAKE_ROWS, SignalRebuildMode.SNAKE_ROWS):
        return _snake_row_indices(height, width, channels)

    if scan_mode in (SignalScanMode.SNAKE_COLUMNS, SignalRebuildMode.SNAKE_COLUMNS):
        return _snake_column_indices(height, width, channels)

    if scan_mode in (
        SignalScanMode.CHANNEL_SEPARATE,
        SignalRebuildMode.CHANNEL_SEPARATE,
    ):
        return _channel_separate_indices(height, width, channels)

    raise ValueError(f"Unsupported scan mode for flat scan indices: {scan_mode}.")


def clear_scan_index_cache() -> None:
    """Clear cached scan/rebuild indices."""
    scan_indices.cache_clear()


def _resolve_rebuild_mode(
    scan_mode: SignalScanMode,
    rebuild_mode: SignalRebuildMode,
) -> SignalRebuildMode:
    if rebuild_mode is not SignalRebuildMode.MATCH_SCAN:
        return rebuild_mode

    if scan_mode in (
        SignalScanMode.ROWS_INDEPENDENT,
        SignalScanMode.COLUMNS_INDEPENDENT,
    ):
        raise ValueError(
            f"{scan_mode} is handled directly by SignalFilterBase and cannot be "
            "used with signal_to_image()."
        )

    return SignalRebuildMode(scan_mode.value)


def _flat_index(
    row: int,
    column: int,
    channel: int,
    width: int,
    channels: int,
) -> int:
    return ((row * width) + column) * channels + channel


def _row_major_indices(
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    return np.arange(height * width * channels, dtype=np.int64)


def _column_major_indices(
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    indices: list[int] = []

    for column in range(width):
        for row in range(height):
            for channel in range(channels):
                indices.append(_flat_index(row, column, channel, width, channels))

    return np.asarray(indices, dtype=np.int64)


def _snake_row_indices(
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    indices: list[int] = []

    for row in range(height):
        if row % 2 == 0:
            columns = range(width)
        else:
            columns = range(width - 1, -1, -1)

        for column in columns:
            for channel in range(channels):
                indices.append(_flat_index(row, column, channel, width, channels))

    return np.asarray(indices, dtype=np.int64)


def _snake_column_indices(
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    indices: list[int] = []

    for column in range(width):
        if column % 2 == 0:
            rows = range(height)
        else:
            rows = range(height - 1, -1, -1)

        for row in rows:
            for channel in range(channels):
                indices.append(_flat_index(row, column, channel, width, channels))

    return np.asarray(indices, dtype=np.int64)


def _channel_separate_indices(
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    indices: list[int] = []

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                indices.append(_flat_index(row, column, channel, width, channels))

    return np.asarray(indices, dtype=np.int64)
