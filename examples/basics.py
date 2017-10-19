import pyglitch.core as pgc
import pyglitch.image_manipulation as pim

# this example shows how to load/save and plot an image and how to perform basic operations on it

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
