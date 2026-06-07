from pyglitch.image import GlitchImage
from pyglitch.glitch import (
    Axis,
    GlitchPipeline,
    Posterize,
    ShiftChannel,
    RescaleImage,
)
from pyglitch.glitch.signal import (
    SignalFlanger,
    SignalReverb,
    SignalScanMode,
    SignalTremolo,
    SignalOutputMode,
    SignalRebuildMode
)

image = GlitchImage.load("tests/image.jpg")

pipeline = GlitchPipeline(
    filters=(
        # ShiftChannel(
        #     channel=0,
        #     offset=24,
        #     axis=Axis.HORIZONTAL,
        # ),
        # SignalReverb(
        #     delay_pixels=32,
        #     decay=0.45,
        #     feedback=True,
        #     scan_mode=SignalScanMode.ROWS_INDEPENDENT,
        #     live_scale=0.5,
        #     mix=0.7,
        # ),
        # SignalFlanger(
        #     max_time_delay=0.003,
        #     rate=0.75,
        #     wet=0.6,
        #     live_scale=0.35,
        #     mix=1.0,
        #     rebuild_mode=SignalRebuildMode.MATCH_SCAN,
        #     output_mode=SignalOutputMode.CLIP,
        # ),
        SignalTremolo(
            frequency=8.0,
            depth=0.35,
            scan_mode=SignalScanMode.ROWS_INDEPENDENT,
            live_scale=0.5,
            mix=0.5,
        ),
        # RescaleImage(),
        # Posterize(bins=12),
        
    )
)

output = pipeline.apply(image)
output.save("tests/live_pipeline.png")