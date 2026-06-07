from __future__ import annotations

from enum import Enum


class SignalScanMode(str, Enum):
    """How an image is read into a 1D signal."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    SNAKE_ROWS = "snake_rows"
    SNAKE_COLUMNS = "snake_columns"
    CHANNEL_SEPARATE = "channel_separate"

    # Live/approximate modes.
    #
    # These process many smaller signals instead of one full-image signal.
    # They are useful for scanline-style effects, easier parallelization later,
    # and more forgiving live visuals.
    ROWS_INDEPENDENT = "rows_independent"
    COLUMNS_INDEPENDENT = "columns_independent"


class SignalRebuildMode(str, Enum):
    """How a processed 1D signal is written back into an image."""

    MATCH_SCAN = "match_scan"
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    SNAKE_ROWS = "snake_rows"
    SNAKE_COLUMNS = "snake_columns"
    CHANNEL_SEPARATE = "channel_separate"


class SignalOutputMode(str, Enum):
    """How processed float signal values are converted back to uint8."""

    CLIP = "clip"
    RESCALE = "rescale"
