from __future__ import annotations

from dataclasses import dataclass, field

from pyglitch.glitch.base import GlitchFilter
from pyglitch.glitch.temporal import (
    TemporalRevealMode,
    blend_images,
    reveal_images,
)
from pyglitch.image.glitch_image import GlitchImage


@dataclass
class FilterTransitionRenderer:
    """Render a filter transition over multiple frames.

    The target image is computed once when start() is called. Subsequent calls to
    next_frame() only interpolate/reveal cached image data, which is useful for
    live control and visual performance.

    Typical usage:

        renderer = FilterTransitionRenderer(
            filter=SignalFlanger(...),
            frames=24,
            mode=TemporalRevealMode.LEFT_TO_RIGHT,
        )

        renderer.start(source_image)

        while not renderer.done:
            frame = renderer.next_frame()
            display(frame)
    """

    filter: GlitchFilter
    frames: int = 24
    mode: TemporalRevealMode = TemporalRevealMode.CROSSFADE
    block_size: int = 32
    random_seed: int = 0

    _source: GlitchImage | None = field(default=None, init=False, repr=False)
    _target: GlitchImage | None = field(default=None, init=False, repr=False)
    _frame_index: int = field(default=0, init=False)

    def start(self, image: GlitchImage) -> None:
        if self.frames <= 0:
            raise ValueError(f"frames must be positive, got {self.frames}.")

        self._source = image.copy()
        self._target = self.filter.apply(image)
        self._frame_index = 0

    @property
    def done(self) -> bool:
        return self._source is not None and self._frame_index >= self.frames

    @property
    def progress(self) -> float:
        if self._source is None:
            return 0.0

        if self.frames <= 1:
            return 1.0

        return min(1.0, self._frame_index / float(self.frames - 1))

    def next_frame(self) -> GlitchImage:
        if self._source is None or self._target is None:
            raise RuntimeError("Transition has not been started. Call start(image).")

        progress = self.progress

        if self.mode is TemporalRevealMode.CROSSFADE:
            data = blend_images(
                original=self._source.data,
                target=self._target.data,
                progress=progress,
            )
        else:
            data = reveal_images(
                original=self._source.data,
                target=self._target.data,
                progress=progress,
                mode=self.mode,
                block_size=self.block_size,
                random_seed=self.random_seed,
            )

        frame = self._source.with_data(data)
        frame.frame_index = self._source.frame_index + self._frame_index

        self._frame_index += 1

        return frame

    def render_all(self) -> list[GlitchImage]:
        return [self.next_frame() for _ in range(self.frames)]
