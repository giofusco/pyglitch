import cv2
import numpy as np

PADDING_RANDOM = 100
PADDING_CIRCULAR = 101
PADDING_COLOR = 102

DIRECTION_LEFT = 104
DIRECTION_RIGHT = 105
DIRECTION_UP = 106
DIRECTION_DOWN = 107


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
    roi = None
    if check_block_boundaries(I, x, y, w, h):
        roi = I[y:y+h, x:x+w].copy()
    return roi


def get_patch(I, x, y, w, h):
    patch = None
    roi = extract_block(I, x, y, w, h)
    if roi is not None:
        patch = (int(x), int(y), int(w), int(h), roi)
    return list(patch)


def swap_patches(I, patch1, patch2):
    I[patch1[1]:patch1[1]+patch1[3], patch1[0]:patch1[0]+patch1[2]] = patch2[4]
    I[patch2[1]:patch2[1] + patch2[3], patch2[0]:patch2[0] + patch2[2]] = patch1[4]
    return I


def shift_rows(I, start, num_rows, offset, padding, color=(0,0,0)):
    pad = None
    if padding == PADDING_CIRCULAR:
        rows = np.roll(I[start:start+num_rows-1,0:-1], offset, axis=1)
        I[start:start + num_rows - 1, 0:-1] = rows
    return I


def swap_channels_at(I, x, y, w, h, channel_idxs):
    patch = get_patch(I, x, y, w, h)
    if patch is not None:
        swap_channels(patch[4], channel_idxs)
        # patch_swapped = (int(x), int(y), int(w), int(h), img_swapped)
        put_patch_in_place(I, patch)
    return I


def swap_channels(I, channel_idxs):
    c0 = I[:-1, :-1, channel_idxs[0]].copy()
    c1 = I[:-1, :-1, channel_idxs[1]].copy()
    c2 = I[:-1, :-1, channel_idxs[2]].copy()
    I[:-1, :-1, 0] = c0
    I[:-1, :-1, 1] = c1
    I[:-1, :-1, 2] = c2
    return I


def shift_channel_hor_at(I, x, y, w, h, channel_idx, offset):
    patch = get_patch(I, x, y, w, h)
    if patch is not None:
        shift_channel_hor(patch[4], channel_idx, offset)
        put_patch_in_place(I, patch)
    return I


def shift_channel_hor(I, channel_idx, offset):
    rows = np.roll(I[:-1,:-1,channel_idx], offset, axis=1)
    I[:-1, :-1, channel_idx] = rows
    return I


def saturate_channel_at(I, x, y, w, h, channel_idx):
    patch = get_patch(I, x, y, w, h)
    if patch is not None:
        saturate_channel(patch[4], channel_idx)
        put_patch_in_place(I, patch)
    return I


def saturate_channel(I, channel_idx):
    I[:-1,:-1,channel_idx] = 255


def put_patch_in_place(I, patch):
    I[patch[1]:patch[1] + patch[3], patch[0]:patch[0] + patch[2]] = patch[4].copy()


def width(I):
    return I.shape[1]


def height(I):
    return I.shape[0]


def num_channels(I):
    return I.shape[2]