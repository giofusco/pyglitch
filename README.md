# Pyglitch

Pyglitch is a basic all Python (>3.5) library to create glitch art from images.
It's released under the Apache 2.0 license.

## Requirements
- numpy==1.13.3
- numba==0.35.0
- matplotlib==2.1.0
- Pillow==4.3.0

## Pyglitch structure

Pyglitch has 3 main components:
- Core: basic functions (e.g. load, visualize and save images)
- Image_Manipulation: this file contains functions to manipulate the image at pixel level
- Audio_Filtering: this contains basic audio filters (reverb, flanger, etc.)


## Getting Started
The folder Examples contains scripts that showcase the main functions available in PyGlitch (more examples will become available in the future)

This simple example should be good enough to get you started
```sh
import pyglitch.core as pgc
import pyglitch.image_manipulation as pim

# load the image
img = pgc.open_image('./images/lena.bmp')
# change the order of the color channels in the image
img = pim.swap_channels(img, [pgc.CH_GREEN, pgc.CH_BLUE, pgc.CH_RED])
# apply posterization
img = pim.posterize(img, 5)
#plot image
pgc.plot_image(img, "Output Image")
#save to disk
pgc.save_image(img, "output.bmp")
```

## Documentation
At the moment the documentation is almost non-existent (sorry!).

