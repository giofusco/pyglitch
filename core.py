# from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# color channel codes
CH_RED = 0
CH_GREEN = 1
CH_BLUE = 2


def rotate_left(I):
    I = np.rot90(I, axes=(1, 0))
    return I


def rotate_right(I):
    I = np.rot90(I, axes=(0, 1))
    return I

def check_block_boundaries(I, x, y, w, h):
    """checks that the rectangle falls withing the image size.
       :param I: numpy array (OpenCV Image)
       :param x: column coord
       :param y: row coord
       :param w: width of the block
       :param h: height of the block
       :return: true if block is valid, false otherwise
    """
    im_size = I.shape
    if x < 0 or y < 0 or x+w >= im_size[1] or y+h >= im_size[0]:
        return False
    else:
        return True


def extract_block(I, x, y, w, h):
    """extract a rectangular region from the image.
       :param I: numpy array (OpenCV Image)
       :param x: column coord
       :param y: row coord
       :param w: width of the block
       :param h: height of the block
       :return: roi, an image region if the boundaries are correct, None otherwise
    """
    roi = None
    if check_block_boundaries(I, x, y, w, h):
        roi = I[y:y+h, x:x+w].copy()
    return roi


def get_patch(I, x, y, w, h):
    """extracts a patch from the image. A patch is a rectangular region from the image combined with its top-left
       corner coordinate within the image and its width and height.
           :param I: numpy array (OpenCV Image)
           :param x: column coord
           :param y: row coord
           :param w: width of the block
           :param h: height of the block
           :return: roi, an image patch if the boundaries are correct, None otherwise
        """
    patch = None
    roi = extract_block(I, x, y, w, h)
    if roi is not None:
        patch = (int(x), int(y), int(w), int(h), roi)
    return list(patch)


def swap_patches(I, patch1, patch2):
    """swaps the content of two patches within the image
        :param I: input/output image
        :param patch1: image patch
        :param patch2: image patch
        :return: modified image I
    """
    I[patch1[1]:patch1[1]+patch1[3], patch1[0]:patch1[0]+patch1[2]] = patch2[4]
    I[patch2[1]:patch2[1] + patch2[3], patch2[0]:patch2[0] + patch2[2]] = patch1[4]
    return I


def plot_image(I, window_name="Output"):
    plt.imshow(I)
    plt.show()


# todo: check that file exists
def open_image(filename, format_ext):
    I = plt.imread(filename, format=format_ext)
    return I


def save_image(I, filename):
    mpimg.imsave(filename, I)


def put_patch_in_place(I, patch):
    """overwrites the location of the patch in the input image using the data contained in patch"""
    I[patch[1]:patch[1] + patch[3], patch[0]:patch[0] + patch[2]] = patch[4].copy()


def to_1d_array(I):
    return np.array(I).ravel()


def width(I):
    """width of image"""
    return I.shape[1]


def height(I):
    """height of image"""
    return I.shape[0]


def num_channels(I):
    """number of channels in the image"""
    return I.shape[2]
