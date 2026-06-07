from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any


class ParameterSmoothing(str, Enum):
    STEP = "step"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ParameterImpact(str, Enum):
    RENDER = "render"
    PRESENTATION = "presentation"
    ENGINE = "engine"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    smoothing: ParameterSmoothing = ParameterSmoothing.LINEAR
    smoothing_seconds: float = 0.08
    impact: ParameterImpact = ParameterImpact.RENDER
    integer: bool = False

    def clamp(self, value: Any) -> Any:
        if not isinstance(value, (int, float)):
            return value
        result = float(value)
        if self.minimum is not None:
            result = max(self.minimum, result)
        if self.maximum is not None:
            result = min(self.maximum, result)
        if self.integer:
            return int(round(result))
        return result


@dataclass
class AnimatedParameter:
    definition: ParameterDefinition
    current: Any
    target: Any
    start_value: Any
    start_time: float
    target_time: float

    @classmethod
    def create(cls, definition: ParameterDefinition, now: float | None = None) -> "AnimatedParameter":
        if now is None:
            now = time.perf_counter()
        value = definition.clamp(definition.default)
        return cls(definition, value, value, value, now, now)

    def set_target(self, value: Any, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        self.current = self.sample(now)
        self.start_value = self.current
        self.target = self.definition.clamp(value)
        self.start_time = now
        self.target_time = now + max(0.0, self.definition.smoothing_seconds)

    def sample(self, now: float | None = None) -> Any:
        if now is None:
            now = time.perf_counter()

        if self.definition.smoothing is ParameterSmoothing.STEP:
            self.current = self.target
            return self.current

        if not _is_numeric(self.start_value) or not _is_numeric(self.target):
            self.current = self.target
            return self.current

        if self.definition.smoothing_seconds <= 0:
            value = self.target
        elif self.definition.smoothing is ParameterSmoothing.LINEAR:
            value = self._sample_linear(now)
        elif self.definition.smoothing is ParameterSmoothing.EXPONENTIAL:
            value = self._sample_exponential(now)
        else:
            raise ValueError(f"Unsupported smoothing mode: {self.definition.smoothing}")

        self.current = self.definition.clamp(value)
        return self.current

    def _sample_linear(self, now: float) -> float:
        if now >= self.target_time:
            return float(self.target)
        duration = self.target_time - self.start_time
        if duration <= 0:
            return float(self.target)
        t = min(1.0, max(0.0, (now - self.start_time) / duration))
        return float(self.start_value) + t * (float(self.target) - float(self.start_value))

    def _sample_exponential(self, now: float) -> float:
        dt = max(0.0, now - self.start_time)
        half_life = max(1e-6, self.definition.smoothing_seconds)
        alpha = 1.0 - math.exp(-0.69314718056 * dt / half_life)
        return float(self.start_value) + alpha * (float(self.target) - float(self.start_value))


@dataclass
class ParameterScheduler:
    parameters: dict[str, AnimatedParameter] = field(default_factory=dict)
    dirty_render: bool = False
    dirty_presentation: bool = False
    dirty_engine: bool = False
    dirty_structural: bool = False

    @classmethod
    def from_definitions(
        cls,
        definitions: tuple[ParameterDefinition, ...],
        now: float | None = None,
    ) -> "ParameterScheduler":
        scheduler = cls()
        for definition in definitions:
            scheduler.add(definition, now)
        return scheduler

    def add(self, definition: ParameterDefinition, now: float | None = None) -> None:
        self.parameters[definition.name] = AnimatedParameter.create(definition, now)

    def set_target(self, name: str, value: Any, now: float | None = None) -> None:
        parameter = self.parameters[name]
        parameter.set_target(value, now)
        self._mark_dirty(parameter.definition.impact)

    def set_targets(self, values: dict[str, Any], now: float | None = None) -> None:
        for name, value in values.items():
            self.set_target(name, value, now)

    def sample(self, now: float | None = None) -> dict[str, Any]:
        return {name: parameter.sample(now) for name, parameter in self.parameters.items()}

    def clear_dirty(self) -> None:
        self.dirty_render = False
        self.dirty_presentation = False
        self.dirty_engine = False
        self.dirty_structural = False

    def _mark_dirty(self, impact: ParameterImpact) -> None:
        if impact is ParameterImpact.RENDER:
            self.dirty_render = True
        elif impact is ParameterImpact.PRESENTATION:
            self.dirty_presentation = True
        elif impact is ParameterImpact.ENGINE:
            self.dirty_engine = True
        elif impact is ParameterImpact.STRUCTURAL:
            self.dirty_structural = True
        else:
            raise ValueError(f"Unsupported parameter impact: {impact}")


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))
