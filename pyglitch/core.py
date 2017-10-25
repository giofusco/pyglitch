#  Copyright 2017 Giovanni Fusco
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Author: Giovanni Fusco - giofusco@gmail.com
#

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#: color channel codes
CH_RED = 0
CH_GREEN = 1
CH_BLUE = 2


def rotate_left(I):
    """ rotate the image 90 degrees to the left"""
    I = np.rot90(I, axes=(1, 0))
    return I


def rotate_right(I):
    """ rotate the image 90 degrees to the right"""
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
    """ plot the image in window
        :param I: the image to plot
        :param window_name: the title of the window of the plot
    """
    plt.imshow(I)
    plt.show()


# todo: check that file exists
def open_image(filename):
    """loads the image
        :param filename: path to the image
    """
    I = plt.imread(filename, format=None)
    if I.dtype == np.float32:
        I *= 255
    return I.copy()


def save_image(I, filename):
    """save image to file
        :param I: image to save
        :param filename: filename where the image will be saved
    """
    mpimg.imsave(filename, I)


def put_patch_in_place(I, patch):
    """overwrites the location of the patch in the input image using the data contained in patch"""
    I[patch[1]:patch[1] + patch[3], patch[0]:patch[0] + patch[2]] = patch[4].copy()


def to_1d_array(I):
    """
    transforms the image from a 3D matrix to a vector
        :param I: input matrix
        :return: 1D vector
    """
    return np.array(I).ravel()


def width(I):
    """returns the width of image"""
    return I.shape[1]


def height(I):
    """returns the height of image"""
    return I.shape[0]


def num_channels(I):
    """number of channels in the image"""
    return I.shape[2]
