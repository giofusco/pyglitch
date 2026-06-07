from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from pyglitch.image.glitch_image import GlitchImage
from pyglitch.live.compositor import FrameCompositor
from pyglitch.live.engine import LiveRenderEngine
from pyglitch.live.parameters import ParameterScheduler

InputCallback = Callable[[ParameterScheduler, float], None]


@dataclass
class LiveRenderPlayer:
    scheduler: ParameterScheduler
    engine: LiveRenderEngine
    compositor: FrameCompositor = field(default_factory=FrameCompositor)
    _last_time: float | None = field(default=None, init=False)

    def start(self) -> GlitchImage:
        now = time.perf_counter()
        params = self.scheduler.sample(now)
        target = self.engine.update(params, now=now, force=True)
        self.compositor.reset(target)
        self._last_time = now
        self.scheduler.clear_dirty()
        return target

    def tick(self, input_callback: InputCallback | None = None) -> GlitchImage:
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
        dt = now - self._last_time
        self._last_time = now

        if input_callback is not None:
            input_callback(self.scheduler, now)

        params = self.scheduler.sample(now)
        target = self.engine.update(params=params, now=now, force=self.scheduler.dirty_structural)
        frame = self.compositor.compose(target=target, dt=dt)
        self.scheduler.clear_dirty()
        return frame

    @property
    def latest_target(self) -> GlitchImage | None:
        return self.engine.latest_target
