from pyglitch.image import GlitchImage, Rect
import pyglitch.glitch as pg

image = GlitchImage.load("tests/image.jpg")

first_rect = Rect(x=100, y=50, width=64, height=64)
second_rect = Rect(x=300, y=120, width=64, height=64)

first_patch = image.get_patch(first_rect)
second_patch = image.get_patch(second_rect)

image.swap_patches(first_patch, second_patch)

  
image.save("tests/output.jpg")


from pyglitch.glitch import (
    Axis,
    GlitchPipeline,
    PixelSortBrightSegments,
    Posterize,
    ShiftChannel,
    SwapRects,
)

pipeline = GlitchPipeline(
    filters=(
        ShiftChannel(channel=0, offset=24, axis=Axis.HORIZONTAL),
        Posterize(bins=8),
        PixelSortBrightSegments(
            red_threshold=128,
            green_threshold=128,
            blue_threshold=128,
        ),
        SwapRects(
            first_rect=Rect(x=10, y=20, width=80, height=80),
            second_rect=Rect(x=200, y=120, width=80, height=80),
        )
    )
)

result = pipeline.apply(image)
result.save("tests/output2.jpg")
