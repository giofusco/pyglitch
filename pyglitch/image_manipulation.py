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

import numpy as np
import colorsys
import pyglitch.core as pgc
from numba import jit, prange


PADDING_RANDOM = 100
PADDING_CIRCULAR = 101
PADDING_COLOR = 102
DIRECTION_LEFT = 104
DIRECTION_RIGHT = 105
DIRECTION_UP = 106
DIRECTION_DOWN = 107


# TODO: handle different kinds of paddings
def shift_rows(X, start, num_rows, offset, padding, color=(0,0,0)):
    I = X.copy()
    pad = None
    if padding == PADDING_CIRCULAR:
        rows = np.roll(I[start:start+num_rows-1,0:-1], offset, axis=1)
        I[start:start + num_rows - 1, 0:-1] = rows
    return I


def swap_channels_at(X, x, y, w, h, channel_idxs):
    I = X.copy()
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        swap_channels(patch[4], channel_idxs)
        # patch_swapped = (int(x), int(y), int(w), int(h), img_swapped)
        pgc.put_patch_in_place(I, patch)
    return I


def swap_channels(X, channel_idxs):
    I = X.copy()
    c0 = I[:-1, :-1, channel_idxs[0]].copy()
    c1 = I[:-1, :-1, channel_idxs[1]].copy()
    c2 = I[:-1, :-1, channel_idxs[2]].copy()
    I[:-1, :-1, 0] = c0
    I[:-1, :-1, 1] = c1
    I[:-1, :-1, 2] = c2
    return I


def shift_channel_hor_at(X, x, y, w, h, channel_idx, offset):
    I = X.copy()
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        shift_channel_hor(patch[4], channel_idx, offset)
        pgc.put_patch_in_place(I, patch)
    return I


def shift_channel_hor(X, channel_idx, offset):
    I = X.copy()
    rows = np.roll(I[:-1,:-1,channel_idx], offset, axis=1)
    I[:-1, :-1, channel_idx] = rows
    return I


def shift_channel_ver(X, channel_idx, offset):
    I = X.copy()
    cols = np.roll(I[:-1,:-1,channel_idx], offset, axis=0)
    I[:-1, :-1, channel_idx] = cols
    return I


def shift_channel_ver_at(X, x, y, w, h, channel_idx, offset):
    I = X.copy()
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        shift_channel_ver(patch[4], channel_idx, offset)
        pgc.put_patch_in_place(I, patch)
    return I


def saturate_channel_at(X, x, y, w, h, channel_idx):
    I = X.copy()
    """saturate a channel in a specific image patch"""
    patch = pgc.get_patch(I, x, y, w, h)
    if patch is not None:
        I = saturate_channel(patch[4], channel_idx)
        pgc.put_patch_in_place(I, patch)
    return I


def saturate_channel(X, channel_idx):
    I = X.copy()
    """"saturates a channel of the image"""
    I = set_channel_value(I, channel_idx, 255)
    return I


def set_channel_value(X, channel_idx, value):
    I = X.copy()
    if channel_idx < pgc.num_channels(I):
        I[:-1,:-1,channel_idx] = value
    return I


def posterize(X, num_bins, normalize=True):
    I = X.copy()
    if normalize:
        I = rescale_image(I)
    bin_size = int(255/num_bins)
    I = (255 * np.round(I/255*num_bins)/num_bins)
    return I.astype(np.uint8)


def pixel_sort_brighter_than_rgb_vert(X, r, g, b, strict=False, sorting_order=('h', 'l', 'v'), iterations=8):
    I = pixel_sort_brighter_than_rgb(pgc.rotate_right(X), r, g, b,
                                     strict=False, sorting_order=('h', 'l', 'v'), iterations=8)
    I = pgc.rotate_left(I)
    return I


@jit
def pixel_sort_brighter_than_rgb(X, r, g, b, strict=False, sorting_order=('h', 'l', 'v'), iterations=8):
    """begin sorting when it finds a pixel which is not (r,g,b) in the column or row,
        and will stop sorting when it finds a (r,g,b) pixel"""
    I = X.copy()
    for row in prange(0, pgc.height(I)):
        if strict:
            from_idx = np.argwhere((I[row, :-1, pgc.CH_RED] > r) & (I[row, :-1, pgc.CH_GREEN] > g)
                                   & (I[row, :-1, pgc.CH_BLUE] > b))
        else:
            from_idx = np.argwhere((I[row, :-1, pgc.CH_RED] > r) | (I[row, :-1, pgc.CH_GREEN] > g)
                                   | (I[row, :-1, pgc.CH_BLUE] > b))

        to_idx = np.argwhere((I[row, :-1, pgc.CH_RED] <= r) & (I[row, :-1, pgc.CH_GREEN] <= g)
                             & (I[row, :-1, pgc.CH_BLUE] <= b))

        if from_idx.size > 0 and to_idx.size > 0:
            i = from_idx[0][0]
            matches = np.argwhere(to_idx > i)
            while not matches.size == 0:
                j = to_idx[matches[0][0]][0]
                I_hlv = _rgb2hlv(I[row, i:j, 0:3], iterations)
                sort_idx = np.argsort(I_hlv, order=sorting_order)
                I[row, i:j] = I[row, i+sort_idx]
                matches_i = np.argwhere(from_idx > j)
                if matches_i.size == 0:
                    break
                else:
                    i = from_idx[matches_i[0][0]][0]
                    matches = np.argwhere(to_idx > i)
    return I


# not sure it's necessary
def rescale_image_rgb(X):
    I = X.copy()
    I_r = rescale_image(I[:,:,0])
    I_g = rescale_image(I[:, :, 1])
    I_b = rescale_image(I[:, :, 2])
    I = np.dstack([I_r, I_g, I_b])
    return  I.astype(np.uint8)


def rescale_image(X):
    I = X.copy()
    if (I.min() < 0 or I.max() > 255):
        I = (I - I.min()) * (255 / (I.max() - I.min()))
        I = I.round()
    return I.astype(np.uint8)



# from http://www.alanzucconi.com/2015/09/30/colour-sorting/
@jit
def _rgb2hlv(I_rgb, repetitions=1):
    I_hlv = []
    for p in prange(0,len(I_rgb)):
        r = I_rgb[p, pgc.CH_RED]
        g = I_rgb[p, pgc.CH_GREEN]
        b = I_rgb[p, pgc.CH_BLUE]

        lum = np.sqrt(.241 * r + .691 * g + .068 * b)

        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        h2 = int(h * repetitions)
        lum2 = int(lum * repetitions)
        v2 = int(v * repetitions)

        if h2 % 2 == 1:
            v2 = repetitions - v2
            lum = repetitions - lum2

        I_hlv.append((h2, lum, v2))

    return np.array(I_hlv, dtype=[('h', '<i4'), ('l', '<i4'), ('v', '<i4')])
