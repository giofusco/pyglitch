from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pygame

# Allow running from the repository root:
#
#     python tools/keyboard_live_demo.py ./images/sf.jpg
#
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyglitch.image import GlitchImage
from pyglitch.glitch import Axis, GlitchPipeline, Posterize, ShiftChannel
from pyglitch.glitch.signal import (
    SignalFlanger,
    SignalOutputMode,
    SignalReverb,
    SignalScanMode,
    SignalTremolo,
)
from pyglitch.live import (
    FrameBlendMode,
    FrameCompositor,
    LiveRenderEngine,
    LiveRenderPlayer,
    ParameterDefinition,
    ParameterImpact,
    ParameterScheduler,
    ParameterSmoothing,
)


@dataclass(frozen=True)
class KeyBinding:
    key: int
    parameter: str
    delta: float
    label: str


KEY_BINDINGS: tuple[KeyBinding, ...] = (
    KeyBinding(pygame.K_LEFT, "shift.offset", -4, "Left: shift -"),
    KeyBinding(pygame.K_RIGHT, "shift.offset", 4, "Right: shift +"),

    KeyBinding(pygame.K_q, "reverb.delay", -4, "Q/W: reverb delay"),
    KeyBinding(pygame.K_w, "reverb.delay", 4, "Q/W: reverb delay"),
    KeyBinding(pygame.K_a, "reverb.decay", -0.03, "A/S: reverb decay"),
    KeyBinding(pygame.K_s, "reverb.decay", 0.03, "A/S: reverb decay"),
    KeyBinding(pygame.K_z, "reverb.mix", -0.03, "Z/X: reverb mix"),
    KeyBinding(pygame.K_x, "reverb.mix", 0.03, "Z/X: reverb mix"),

    KeyBinding(pygame.K_e, "flanger.delay", -0.00025, "E/R: flanger delay"),
    KeyBinding(pygame.K_r, "flanger.delay", 0.00025, "E/R: flanger delay"),
    KeyBinding(pygame.K_d, "flanger.wet", -0.03, "D/F: flanger wet"),
    KeyBinding(pygame.K_f, "flanger.wet", 0.03, "D/F: flanger wet"),
    KeyBinding(pygame.K_c, "flanger.mix", -0.03, "C/V: flanger mix"),
    KeyBinding(pygame.K_v, "flanger.mix", 0.03, "C/V: flanger mix"),

    KeyBinding(pygame.K_t, "tremolo.frequency", -0.5, "T/Y: tremolo freq"),
    KeyBinding(pygame.K_y, "tremolo.frequency", 0.5, "T/Y: tremolo freq"),
    KeyBinding(pygame.K_g, "tremolo.depth", -0.03, "G/H: tremolo depth"),
    KeyBinding(pygame.K_h, "tremolo.depth", 0.03, "G/H: tremolo depth"),
    KeyBinding(pygame.K_b, "tremolo.mix", -0.03, "B/N: tremolo mix"),
    KeyBinding(pygame.K_n, "tremolo.mix", 0.03, "B/N: tremolo mix"),

    KeyBinding(pygame.K_1, "posterize.bins", -1, "1/2: posterize bins"),
    KeyBinding(pygame.K_2, "posterize.bins", 1, "1/2: posterize bins"),

    KeyBinding(pygame.K_3, "engine.render_fps", -1, "3/4: render FPS"),
    KeyBinding(pygame.K_4, "engine.render_fps", 1, "3/4: render FPS"),

    KeyBinding(pygame.K_5, "display.half_life", -0.01, "5/6: display smoothing"),
    KeyBinding(pygame.K_6, "display.half_life", 0.01, "5/6: display smoothing"),
)


def parameter_definitions() -> tuple[ParameterDefinition, ...]:
    return (
        ParameterDefinition(
            name="shift.offset",
            default=0,
            minimum=-220,
            maximum=220,
            smoothing=ParameterSmoothing.LINEAR,
            smoothing_seconds=0.10,
            integer=True,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="reverb.delay",
            default=32,
            minimum=1,
            maximum=512,
            smoothing=ParameterSmoothing.LINEAR,
            smoothing_seconds=0.10,
            integer=True,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="reverb.decay",
            default=0.18,
            minimum=0.0,
            maximum=0.8,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.12,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="reverb.mix",
            default=0.35,
            minimum=0.0,
            maximum=1.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.12,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="flanger.delay",
            default=0.0015,
            minimum=0.0,
            maximum=0.01,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.12,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="flanger.rate",
            default=0.75,
            minimum=0.0,
            maximum=8.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.12,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="flanger.wet",
            default=0.35,
            minimum=0.0,
            maximum=1.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.12,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="flanger.mix",
            default=0.35,
            minimum=0.0,
            maximum=1.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.12,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="tremolo.frequency",
            default=6.0,
            minimum=0.0,
            maximum=40.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.10,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="tremolo.depth",
            default=0.18,
            minimum=0.0,
            maximum=1.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.10,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="tremolo.mix",
            default=0.25,
            minimum=0.0,
            maximum=1.0,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.10,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="posterize.bins",
            default=14,
            minimum=2,
            maximum=64,
            smoothing=ParameterSmoothing.STEP,
            smoothing_seconds=0.0,
            integer=True,
            impact=ParameterImpact.RENDER,
        ),
        ParameterDefinition(
            name="engine.render_fps",
            default=20,
            minimum=1,
            maximum=60,
            smoothing=ParameterSmoothing.STEP,
            smoothing_seconds=0.0,
            integer=True,
            impact=ParameterImpact.ENGINE,
        ),
        ParameterDefinition(
            name="display.half_life",
            default=0.08,
            minimum=0.0,
            maximum=0.5,
            smoothing=ParameterSmoothing.EXPONENTIAL,
            smoothing_seconds=0.08,
            impact=ParameterImpact.PRESENTATION,
        ),
    )


def build_pipeline(params: dict) -> GlitchPipeline:
    return GlitchPipeline(
        filters=(
            ShiftChannel(
                channel=0,
                offset=int(params["shift.offset"]),
                axis=Axis.HORIZONTAL,
            ),
            SignalReverb(
                delay_pixels=int(params["reverb.delay"]),
                decay=float(params["reverb.decay"]),
                feedback=True,
                scan_mode=SignalScanMode.ROW_MAJOR,
                live_scale=1.0,
                mix=float(params["reverb.mix"]),
                output_mode=SignalOutputMode.CLIP,
            ),
            SignalFlanger(
                max_time_delay=float(params["flanger.delay"]),
                rate=float(params["flanger.rate"]),
                dry=1.0,
                wet=float(params["flanger.wet"]),
                scan_mode=SignalScanMode.ROW_MAJOR,
                live_scale=1.0,
                mix=float(params["flanger.mix"]),
                output_mode=SignalOutputMode.CLIP,
            ),
            SignalTremolo(
                frequency=float(params["tremolo.frequency"]),
                depth=float(params["tremolo.depth"]),
                scan_mode=SignalScanMode.ROW_MAJOR,
                live_scale=1.0,
                mix=float(params["tremolo.mix"]),
                output_mode=SignalOutputMode.CLIP,
            ),
            Posterize(
                bins=int(params["posterize.bins"]),
                normalize=False,
            ),
        )
    )


def glitch_image_to_surface(image: GlitchImage) -> pygame.Surface:
    rgb = image.data[:, :, :3]
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return surface


def resize_image_for_preview(image: GlitchImage, max_width: int | None) -> GlitchImage:
    if max_width is None or image.width <= max_width:
        return image

    from scipy import ndimage

    scale = max_width / float(image.width)
    height = max(1, int(round(image.height * scale)))
    zoom = (height / image.height, max_width / image.width, 1.0)

    resized = ndimage.zoom(image.data, zoom=zoom, order=1)
    resized = np.clip(resized, 0, 255).round().astype(np.uint8)

    return image.with_data(resized)


def get_param(scheduler: ParameterScheduler, name: str, now: float) -> float:
    return scheduler.parameters[name].sample(now)


def nudge_parameter(
    scheduler: ParameterScheduler,
    name: str,
    delta: float,
    now: float,
) -> None:
    current = scheduler.parameters[name].sample(now)
    scheduler.set_target(name, current + delta, now)


def handle_keydown(
    event: pygame.event.Event,
    scheduler: ParameterScheduler,
    engine: LiveRenderEngine,
    compositor: FrameCompositor,
    now: float,
) -> bool:
    """Return False when app should quit."""
    if event.key == pygame.K_ESCAPE:
        return False

    if event.key == pygame.K_SPACE:
        # Reset all parameters to defaults.
        for name, parameter in scheduler.parameters.items():
            scheduler.set_target(name, parameter.definition.default, now)
        engine.reset()
        return True

    if event.key == pygame.K_m:
        # Toggle hard cut / crossfade display.
        if compositor.mode is FrameBlendMode.CROSSFADE:
            compositor.mode = FrameBlendMode.HARD_CUT
        else:
            compositor.mode = FrameBlendMode.CROSSFADE
        return True

    for binding in KEY_BINDINGS:
        if event.key == binding.key:
            nudge_parameter(scheduler, binding.parameter, binding.delta, now)
            return True

    return True


def update_engine_and_display_settings(
    scheduler: ParameterScheduler,
    engine: LiveRenderEngine,
    compositor: FrameCompositor,
    now: float,
) -> None:
    render_fps = max(1, int(get_param(scheduler, "engine.render_fps", now)))
    engine.min_render_interval = 1.0 / render_fps

    compositor.half_life = float(get_param(scheduler, "display.half_life", now))


def draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    scheduler: ParameterScheduler,
    engine: LiveRenderEngine,
    compositor: FrameCompositor,
    display_fps: float,
    now: float,
) -> None:
    values = scheduler.sample(now)

    lines = [
        "pyglitch keyboard live demo",
        "",
        f"display fps: {display_fps:5.1f}",
        f"render fps cap: {int(values['engine.render_fps'])}",
        f"renders: {engine.render_count}",
        f"compositor: {compositor.mode.value}, half_life={values['display.half_life']:.3f}",
        "",
        f"shift.offset:       {int(values['shift.offset'])}",
        f"reverb.delay:       {int(values['reverb.delay'])}",
        f"reverb.decay:       {values['reverb.decay']:.3f}",
        f"reverb.mix:         {values['reverb.mix']:.3f}",
        f"flanger.delay:      {values['flanger.delay']:.5f}",
        f"flanger.wet:        {values['flanger.wet']:.3f}",
        f"flanger.mix:        {values['flanger.mix']:.3f}",
        f"tremolo.frequency:  {values['tremolo.frequency']:.2f}",
        f"tremolo.depth:      {values['tremolo.depth']:.3f}",
        f"tremolo.mix:        {values['tremolo.mix']:.3f}",
        f"posterize.bins:     {int(values['posterize.bins'])}",
        "",
        "keys:",
        "left/right shift",
        "q/w reverb delay    a/s reverb decay    z/x reverb mix",
        "e/r flanger delay   d/f flanger wet      c/v flanger mix",
        "t/y tremolo freq    g/h tremolo depth    b/n tremolo mix",
        "1/2 posterize bins  3/4 render fps       5/6 display smoothing",
        "m toggle hard/crossfade, space reset, esc quit",
    ]

    x = 10
    y = 10

    for line in lines:
        shadow = font.render(line, True, (0, 0, 0))
        text = font.render(line, True, (255, 255, 255))
        screen.blit(shadow, (x + 1, y + 1))
        screen.blit(text, (x, y))
        y += 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard-controlled pyglitch live demo.")
    parser.add_argument("image", type=str, help="Path to input image.")
    parser.add_argument("--max-width", type=int, default=960, help="Resize image for preview if wider than this.")
    parser.add_argument("--display-fps", type=int, default=30, help="Pygame display FPS.")
    parser.add_argument("--no-overlay", action="store_true", help="Hide text overlay.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    pygame.init()

    image = GlitchImage.load(image_path)
    image = resize_image_for_preview(image, args.max_width)

    scheduler = ParameterScheduler.from_definitions(parameter_definitions())

    engine = LiveRenderEngine(
        source_image=image,
        build_pipeline=build_pipeline,
        min_render_interval=1.0 / 20.0,
    )

    compositor = FrameCompositor(
        mode=FrameBlendMode.CROSSFADE,
        half_life=0.08,
    )

    player = LiveRenderPlayer(
        scheduler=scheduler,
        engine=engine,
        compositor=compositor,
    )

    screen = pygame.display.set_mode((image.width, image.height))
    pygame.display.set_caption("pyglitch keyboard live demo")

    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    player.start()

    running = True

    while running:
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = handle_keydown(
                    event=event,
                    scheduler=scheduler,
                    engine=engine,
                    compositor=compositor,
                    now=now,
                )

        update_engine_and_display_settings(
            scheduler=scheduler,
            engine=engine,
            compositor=compositor,
            now=now,
        )

        frame = player.tick()
        surface = glitch_image_to_surface(frame)
        screen.blit(surface, (0, 0))

        display_fps = clock.get_fps()

        if not args.no_overlay:
            draw_overlay(
                screen=screen,
                font=font,
                scheduler=scheduler,
                engine=engine,
                compositor=compositor,
                display_fps=display_fps,
                now=now,
            )

        pygame.display.flip()
        clock.tick(args.display_fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
