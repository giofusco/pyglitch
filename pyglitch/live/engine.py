from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from typing import Any
from pyglitch.glitch.pipeline import GlitchPipeline
from pyglitch.image.glitch_image import GlitchImage

PipelineBuilder = Callable[[dict[str, Any]], GlitchPipeline]


@dataclass
class LiveRenderEngine:
    source_image: GlitchImage
    build_pipeline: PipelineBuilder
    min_render_interval: float = 1.0 / 30.0
    render_on_every_update: bool = False

    latest_target: GlitchImage | None = field(default=None, init=False)
    last_render_time: float = field(default=0.0, init=False)
    last_params: dict[str, Any] | None = field(default=None, init=False)
    render_count: int = field(default=0, init=False)

    def update(self, params: dict[str, Any], now: float | None = None, force: bool = False) -> GlitchImage:
        if now is None:
            now = time.perf_counter()
        if self.latest_target is None:
            return self.render(params, now)
        if force or self.render_on_every_update:
            return self.render(params, now)
        if now - self.last_render_time < self.min_render_interval:
            return self.latest_target
        if params == self.last_params:
            return self.latest_target
        return self.render(params, now)

    def render(self, params: dict[str, Any], now: float | None = None) -> GlitchImage:
        if now is None:
            now = time.perf_counter()
        pipeline = self.build_pipeline(params)
        self.latest_target = pipeline.apply(self.source_image)
        self.last_render_time = now
        self.last_params = dict(params)
        self.render_count += 1
        return self.latest_target

    def reset(self, source_image: GlitchImage | None = None) -> None:
        if source_image is not None:
            self.source_image = source_image
        self.latest_target = None
        self.last_render_time = 0.0
        self.last_params = None
        self.render_count = 0
