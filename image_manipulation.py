import numpy as np
import pyglitch.core as pgc

PADDING_RANDOM = 100
PADDING_CIRCULAR = 101
PADDING_COLOR = 102
DIRECTION_LEFT = 104
DIRECTION_RIGHT = 105
DIRECTION_UP = 106
DIRECTION_DOWN = 107

# TODO: handle different kinds of paddings
def shift_rows(I, start, num_rows, offset, padding, color=(0,0,0)):
    pad = None
    if padding == PADDING_CIRCULAR:
        rows = np.roll(I[start:start+num_rows-1,0:-1], offset, axis=1)
        I[start:start + num_rows - 1, 0:-1] = rows
    return I


def swap_channels_at(I, x, y, w, h, channel_idxs):
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        swap_channels(patch[4], channel_idxs)
        # patch_swapped = (int(x), int(y), int(w), int(h), img_swapped)
        pgc.put_patch_in_place(I, patch)
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
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        shift_channel_hor(patch[4], channel_idx, offset)
        pgc.put_patch_in_place(I, patch)
    return I


def shift_channel_hor(I, channel_idx, offset):
    rows = np.roll(I[:-1,:-1,channel_idx], offset, axis=1)
    I[:-1, :-1, channel_idx] = rows
    return I


def shift_channel_ver(I, channel_idx, offset):
    cols = np.roll(I[:-1,:-1,channel_idx], offset, axis=0)
    I[:-1, :-1, channel_idx] = cols
    return I


def shift_channel_ver_at(I, x, y, w, h, channel_idx, offset):
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        shift_channel_ver(patch[4], channel_idx, offset)
        pgc.put_patch_in_place(I, patch)
    return I


def saturate_channel_at(I, x, y, w, h, channel_idx):
    """saturate a channel in a specific image patch"""
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        saturate_channel(patch[4], channel_idx)
        pgc.put_patch_in_place(I, patch)
    return I


def saturate_channel(I, channel_idx):
    """"saturates a channel of the image"""
    if channel_idx < pgc.num_channels(I):
        I[:-1,:-1,channel_idx] = 255
    return I
