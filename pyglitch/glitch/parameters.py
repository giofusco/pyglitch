from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParameterKind(str, Enum):
    """Supported public parameter types for UI/MIDI/preset mapping."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    RECT = "rect"
    FILTER = "filter"
    FILTER_TUPLE = "filter_tuple"
    ARRAY = "array"


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata describing one controllable filter or bank parameter.

    This class is intentionally separate from MIDI and UI code. It describes
    the available parameter, its range, its default value, and optional enum
    values.
    """

    name: str
    kind: ParameterKind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    enum_values: tuple[Any, ...] = ()
    label: str | None = None
    description: str | None = None

    def clamp(self, value: Any) -> Any:
        """Clamp numeric values to this parameter range.

        Non-numeric parameter kinds are returned unchanged.
        """
        if self.kind not in (ParameterKind.INT, ParameterKind.FLOAT):
            return value

        numeric_value = float(value)

        if self.minimum is not None:
            numeric_value = max(self.minimum, numeric_value)

        if self.maximum is not None:
            numeric_value = min(self.maximum, numeric_value)

        if self.kind is ParameterKind.INT:
            return int(round(numeric_value))

        return numeric_value


class Parameterized:
    """Mixin for objects that expose public parameter metadata."""

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return ()
