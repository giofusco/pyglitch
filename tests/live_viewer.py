from __future__ import annotations

from dataclasses import dataclass, replace
import sys

import numpy as np
import pygame

from pyglitch.image import GlitchImage
from pyglitch.glitch import (
    Axis,
    GlitchPipeline,
    Posterize,
    ShiftChannel,
)
from pyglitch.glitch.signal import (
    SignalFlanger,
    SignalReverb,
    SignalScanMode,
)
from pyglitch.glitch.temporal import (
    TemporalRevealMode,
    blend_images,
    reveal_images,
)


@dataclass
class LiveState:
    shift_offset: int = 32
    reverb_delay: int = 48
    reverb_decay: float = 0.5
    flanger_delay: float = 0.003
    flanger_rate: float = 0.8
    flanger_wet: float = 0.6
    posterize_bins: int = 10
    reveal_mode: TemporalRevealMode = TemporalRevealMode.LEFT_TO_RIGHT
    transition_frames: int = 18


def build_pipeline(state: LiveState) -> GlitchPipeline:
    return GlitchPipeline(
        filters=(
            ShiftChannel(
                channel=0,
                offset=state.shift_offset,
                axis=Axis.HORIZONTAL,
            ),
            SignalReverb(
                delay_pixels=state.reverb_delay,
                decay=state.reverb_decay,
                feedback=True,
                scan_mode=SignalScanMode.ROWS_INDEPENDENT,
                live_scale=0.5,
                mix=0.8,
            ),
            SignalFlanger(
                max_time_delay=state.flanger_delay,
                rate=state.flanger_rate,
                wet=state.flanger_wet,
                live_scale=0.35,
                mix=0.7,
            ),
            Posterize(
                bins=state.posterize_bins,
            ),
        )
    )


class LivePipelinePlayer:
    """
    Stable live player:
    - target is always computed from the original source image
    - transition starts from the current displayed frame
    """

    def __init__(
        self,
        source_image: GlitchImage,
        initial_pipeline: GlitchPipeline,
        frames: int = 18,
        mode: TemporalRevealMode = TemporalRevealMode.LEFT_TO_RIGHT,
        block_size: int = 32,
        random_seed: int = 0,
    ) -> None:
        self.source_image = source_image
        self.pipeline = initial_pipeline
        self.frames = max(1, frames)
        self.mode = mode
        self.block_size = block_size
        self.random_seed = random_seed

        self.current_frame = source_image.copy()
        self.transition_source = source_image.copy()
        self.transition_target = initial_pipeline.apply(source_image)
        self.frame_index = self.frames  # start "done"

        self.current_frame = self.transition_target.copy()

    def set_pipeline(
        self,
        pipeline: GlitchPipeline,
        frames: int | None = None,
        mode: TemporalRevealMode | None = None,
    ) -> None:
        self.pipeline = pipeline

        if frames is not None:
            self.frames = max(1, frames)

        if mode is not None:
            self.mode = mode

        self.transition_source = self.current_frame.copy()
        self.transition_target = self.pipeline.apply(self.source_image)
        self.frame_index = 0

    def next_frame(self) -> GlitchImage:
        if self.frame_index >= self.frames:
            self.current_frame = self.transition_target.copy()
            return self.current_frame

        if self.frames == 1:
            progress = 1.0
        else:
            progress = self.frame_index / float(self.frames - 1)

        if self.mode is TemporalRevealMode.CROSSFADE:
            data = blend_images(
                original=self.transition_source.data,
                target=self.transition_target.data,
                progress=progress,
            )
        else:
            data = reveal_images(
                original=self.transition_source.data,
                target=self.transition_target.data,
                progress=progress,
                mode=self.mode,
                block_size=self.block_size,
                random_seed=self.random_seed,
            )

        self.current_frame = self.transition_source.with_data(data)
        self.frame_index += 1
        return self.current_frame


def glitch_image_to_surface(image: GlitchImage) -> pygame.Surface:
    rgb = image.data[:, :, :3]
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return surface


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def main() -> int:
    pygame.init()

    image = GlitchImage.load("tests/image.jpg")

    state = LiveState()
    pipeline = build_pipeline(state)

    player = LivePipelinePlayer(
        source_image=image,
        initial_pipeline=pipeline,
        frames=state.transition_frames,
        mode=state.reveal_mode,
        block_size=32,
        random_seed=42,
    )

    screen = pygame.display.set_mode((image.width, image.height))
    pygame.display.set_caption("pyglitch live viewer")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    def restart_transition() -> None:
        nonlocal player, state
        pipeline = build_pipeline(state)
        player.set_pipeline(
            pipeline=pipeline,
            frames=state.transition_frames,
            mode=state.reveal_mode,
        )

    restart_transition()

    running = True

    while running:
        changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = event.key

                if key == pygame.K_ESCAPE:
                    running = False

                elif key == pygame.K_LEFT:
                    state.shift_offset -= 4
                    changed = True
                elif key == pygame.K_RIGHT:
                    state.shift_offset += 4
                    changed = True

                elif key == pygame.K_q:
                    state.reverb_delay -= 4
                    changed = True
                elif key == pygame.K_w:
                    state.reverb_delay += 4
                    changed = True

                elif key == pygame.K_a:
                    state.reverb_decay -= 0.05
                    changed = True
                elif key == pygame.K_s:
                    state.reverb_decay += 0.05
                    changed = True

                elif key == pygame.K_e:
                    state.flanger_delay -= 0.00025
                    changed = True
                elif key == pygame.K_r:
                    state.flanger_delay += 0.00025
                    changed = True

                elif key == pygame.K_d:
                    state.flanger_wet -= 0.05
                    changed = True
                elif key == pygame.K_f:
                    state.flanger_wet += 0.05
                    changed = True

                elif key == pygame.K_z:
                    state.posterize_bins -= 1
                    changed = True
                elif key == pygame.K_x:
                    state.posterize_bins += 1
                    changed = True

                elif key == pygame.K_1:
                    state.reveal_mode = TemporalRevealMode.CROSSFADE
                    changed = True
                elif key == pygame.K_2:
                    state.reveal_mode = TemporalRevealMode.LEFT_TO_RIGHT
                    changed = True
                elif key == pygame.K_3:
                    state.reveal_mode = TemporalRevealMode.TOP_TO_BOTTOM
                    changed = True
                elif key == pygame.K_4:
                    state.reveal_mode = TemporalRevealMode.CENTER_OUT
                    changed = True
                elif key == pygame.K_5:
                    state.reveal_mode = TemporalRevealMode.RANDOM_BLOCKS
                    changed = True

                elif key == pygame.K_MINUS:
                    state.transition_frames -= 2
                    changed = True
                elif key == pygame.K_EQUALS:
                    state.transition_frames += 2
                    changed = True

        # Clamp parameters
        state.shift_offset = clamp_int(state.shift_offset, -200, 200)
        state.reverb_delay = clamp_int(state.reverb_delay, 1, 512)
        state.reverb_decay = clamp_float(state.reverb_decay, 0.0, 1.0)
        state.flanger_delay = clamp_float(state.flanger_delay, 0.0, 0.01)
        state.flanger_wet = clamp_float(state.flanger_wet, 0.0, 1.0)
        state.posterize_bins = clamp_int(state.posterize_bins, 2, 64)
        state.transition_frames = clamp_int(state.transition_frames, 1, 120)

        if changed:
            restart_transition()

        frame = player.next_frame()
        surface = glitch_image_to_surface(frame)

        screen.blit(surface, (0, 0))

        overlay_lines = [
            f"shift: {state.shift_offset}",
            f"reverb delay: {state.reverb_delay}",
            f"reverb decay: {state.reverb_decay:.2f}",
            f"flanger delay: {state.flanger_delay:.5f}",
            f"flanger wet: {state.flanger_wet:.2f}",
            f"posterize bins: {state.posterize_bins}",
            f"transition frames: {state.transition_frames}",
            f"mode: {state.reveal_mode.value}",
            "keys:",
            "left/right shift",
            "q/w reverb delay",
            "a/s reverb decay",
            "e/r flanger delay",
            "d/f flanger wet",
            "z/x posterize bins",
            "1..5 reveal mode",
            "- / = transition length",
            "esc quit",
        ]

        y = 8
        for line in overlay_lines:
            text_surface = font.render(line, True, (255, 255, 255))
            shadow_surface = font.render(line, True, (0, 0, 0))
            screen.blit(shadow_surface, (9, y + 1))
            screen.blit(text_surface, (8, y))
            y += 20

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
