from __future__ import annotations

from pathlib import Path
import math
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyglitch.glitch import Axis, GlitchPipeline, Posterize, ShiftChannel
from pyglitch.glitch.signal import SignalFlanger, SignalOutputMode, SignalReverb, SignalScanMode
from pyglitch.image import GlitchImage
from pyglitch.live import (
    FrameCompositor,
    LiveRenderEngine,
    LiveRenderPlayer,
    ParameterDefinition,
    ParameterImpact,
    ParameterScheduler,
    ParameterSmoothing,
)


def build_pipeline(params: dict) -> GlitchPipeline:
    return GlitchPipeline(
        filters=(
            ShiftChannel(channel=0, offset=int(params["shift.offset"]), axis=Axis.HORIZONTAL),
            SignalReverb(
                delay_pixels=int(params["reverb.delay"]),
                decay=float(params["reverb.decay"]),
                feedback=False,
                scan_mode=SignalScanMode.ROWS_INDEPENDENT,
                live_scale=0.6,
                mix=0.45,
                output_mode=SignalOutputMode.CLIP,
            ),
            SignalFlanger(
                max_time_delay=float(params["flanger.delay"]),
                rate=float(params["flanger.rate"]),
                wet=float(params["flanger.wet"]),
                live_scale=0.45,
                mix=0.45,
                output_mode=SignalOutputMode.CLIP,
            ),
            Posterize(bins=int(params["posterize.bins"]), normalize=True),
        )
    )


def main() -> int:
    image = GlitchImage.load("tools/sf.jpg")
    output_dir = Path("./outputs/live_engine_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    definitions = (
        ParameterDefinition("shift.offset", 0, -160, 160, ParameterSmoothing.LINEAR, 0.15, ParameterImpact.RENDER, True),
        ParameterDefinition("reverb.delay", 32, 1, 256, ParameterSmoothing.LINEAR, 0.12, ParameterImpact.RENDER, True),
        ParameterDefinition("reverb.decay", 0.25, 0.0, 1.0, ParameterSmoothing.EXPONENTIAL, 0.10),
        ParameterDefinition("flanger.delay", 0.001, 0.0, 0.008, ParameterSmoothing.EXPONENTIAL, 0.10),
        ParameterDefinition("flanger.rate", 0.7, 0.0, 4.0, ParameterSmoothing.EXPONENTIAL, 0.10),
        ParameterDefinition("flanger.wet", 0.35, 0.0, 1.0, ParameterSmoothing.EXPONENTIAL, 0.10),
        ParameterDefinition("posterize.bins", 14, 2, 64, ParameterSmoothing.STEP, 0.0, ParameterImpact.RENDER, True),
    )

    scheduler = ParameterScheduler.from_definitions(definitions)
    engine = LiveRenderEngine(image, build_pipeline, min_render_interval=1.0 / 20.0)
    player = LiveRenderPlayer(scheduler, engine, FrameCompositor(half_life=0.08))
    player.start()

    start = time.perf_counter()
    for frame_index in range(90):
        now = time.perf_counter()
        t = now - start
        scheduler.set_targets(
            {
                "shift.offset": int(80.0 * math.sin(t * 1.7)),
                "flanger.wet": 0.25 + 0.25 * (1.0 + math.sin(t * 1.1)),
                "reverb.decay": 0.15 + 0.20 * (1.0 + math.sin(t * 0.8)),
            },
            now=now,
        )

        frame = player.tick()
        frame.save(output_dir / f"frame_{frame_index:03d}.png")
        time.sleep(1.0 / 30.0)

    print(f"Wrote frames to: {output_dir}")
    print(f"Engine rendered {engine.render_count} target frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
