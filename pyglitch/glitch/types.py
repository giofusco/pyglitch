from __future__ import annotations

from enum import Enum


class Axis(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class PixelateOperator(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


class PixelSortKey(str, Enum):
    LUMINANCE = "luminance"
    VALUE = "value"
    HUE = "hue"
