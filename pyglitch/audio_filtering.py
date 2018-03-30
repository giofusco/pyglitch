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

import pyglitch.core as pgc
import numpy as np
import math
from numba import jit, prange


@jit
def reverb(I, delay_pixels, decay = 0.5):
    assert delay_pixels != 0, "delay_pixels must be not zero"
    x = pgc.to_1d_array(I)
    if delay_pixels > 0:
        for i in prange(0, len(x) - delay_pixels-1):
            # WARNING: overflow potential
            # x[i + delay_pixels] += (x[i] * decay).astype(np.uint8)
            x[i + delay_pixels] += (x[i] * decay)
    elif delay_pixels < 0:
        for i in prange(len(x)-1, -delay_pixels+1, -1):
            # WARNING: overflow potential
            # x[i + delay_pixels] += (x[i] * decay).astype(np.uint8)
            x[i + delay_pixels] += (x[i] * decay)
    I = np.reshape(x, I.shape)
    return I
    # return I.astype(np.uint8)


#TODO: optimize code
@jit
def wah_wah(I, damp=0.05, minf=500, maxf=5000, Fw=2000, Fs=44100):
    """wah-wah filter
        :parameter damp: damping factor. lower the damping factor the smaller the pass band
        :parameter minf: min center cut off frequency of variable bandpass filter
        :parameter maxf: max center cutoff frequency of variable bandpass filter
        :parameter Fw: wah frequency, how many Hz per second are cycled through
        :parameter Fs: sampling rate in Hz
    """

    delta = Fw / Fs
    x = pgc.to_1d_array(I)
    Fc = np.arange(minf, maxf, delta)
    while len(Fc) < len(x):
        Fc = np.append(Fc, np.arange(maxf, minf, -delta))
        Fc = np.append(Fc, np.arange(minf, maxf, delta))
    Fc = Fc[1:len(x)]
    F1 = 2 * math.sin((math.pi * Fc[1]) / Fs)
    Q1 = 2 * damp
    yh = np.zeros(len(x))
    yb = np.zeros(len(x))
    yl = np.zeros(len(x))
    yh[1] = x[1]
    yb[1] = F1 * yh[1]
    yl[1] = F1 * yb[1]
    for n in prange(2, len(x)-1):
        yh[n] = x[n] - yl[n - 1] - Q1 * yb[n - 1]
        yb[n] = F1 * yh[n] + yb[n - 1]
        yl[n] = F1 * yb[n] + yl[n - 1]
        F1 = 2 * math.sin((math.pi * Fc[n]) / Fs)
    maxyb = max(abs(yb))
    yb = yb / maxyb
    I = np.reshape(yb, I.shape)
    return I


@jit
def flanger(X, max_time_delay=0.003, rate=1, Fs=44100, amp=0.7):
    I = X.copy()
    if pgc.num_channels(X) == 3:
        I = __flangerRGB(I, max_time_delay, rate, Fs, amp)
        return I
    else:
        x = pgc.to_1d_array(I)
        idx = np.arange(0, len(x))
        sin_ref = (np.sin(2 * math.pi * idx * (rate / Fs)))
        max_samp_delay = round(max_time_delay * Fs)
        y = np.zeros(len(x))
        y[1: max_samp_delay] = x[1: max_samp_delay]
        for i in prange(max_samp_delay+1, len(x)):
            cur_sin = np.abs(sin_ref[i])
            cur_delay = math.ceil(cur_sin * max_samp_delay)
            y[i] = (amp * x[i]) + amp * (x[i - cur_delay])
        I = np.reshape(y, I.shape)
    return I
    # return I.astype(np.uint8)


@jit
def __flangerRGB(I, max_time_delay, rate, Fs, amp):
    x0 = pgc.to_1d_array(I[:,:,0])
    x1 = pgc.to_1d_array(I[:, :, 1])
    x2 = pgc.to_1d_array(I[:, :, 2])
    idx = np.arange(0, len(x0))
    sin_ref = (np.sin(2 * math.pi * idx * (rate / Fs)))
    max_samp_delay = round(max_time_delay * Fs)
    y0 = np.zeros(len(x0))
    y0[1: max_samp_delay] = x0[1: max_samp_delay]
    y1 = np.zeros(len(x0))
    y1[1: max_samp_delay] = x1[1: max_samp_delay]
    y2 = np.zeros(len(x0))
    y2[1: max_samp_delay] = x2[1: max_samp_delay]
    for i in prange(max_samp_delay+1, len(x0)):
        cur_sin = np.abs(sin_ref[i])
        cur_delay = math.ceil(cur_sin * max_samp_delay)
        y0[i] = (amp * x0[i]) + amp * (x0[i - cur_delay])
        y1[i] = (amp * x1[i]) + amp * (x1[i - cur_delay])
        y2[i] = (amp * x2[i]) + amp * (x2[i - cur_delay])

    I[:, :, 0] = np.reshape(y0, (pgc.height(I), pgc.width(I)))
    I[:, :, 1] = np.reshape(y1, (pgc.height(I), pgc.width(I)))
    I[:, :, 2] = np.reshape(y2, (pgc.height(I), pgc.width(I)))

    return I


def tremolo(X, Fc=5, alpha=0.5, Fs=44100):
    # assert (X.shape[2] == 0 or X.shape[2] == 3)
    I = X.copy()
    if pgc.num_channels(X) == 3:
        I = __tremoloRGB(I, Fc, alpha, Fs)
        return I
    else:
        x = pgc.to_1d_array(I)
        index = np.arange(0, len(x))
        trem = (1 + alpha * np.sin(2 * np.pi * index * (Fc / Fs)))
        y = np.multiply(x,trem)
        I = np.reshape(y, I.shape)
        return I
    # return I.astype(np.uint8)


@jit
def __tremoloRGB(I, Fc=5, alpha=0.5, Fs=44100):
    index = np.arange(0, pgc.width(I)*pgc.height(I))
    trem = (1 + alpha * np.sin(2 * np.pi * index * (Fc / Fs)))
    for c in prange(0, 2):
        x = pgc.to_1d_array(I[:, :, c])
        y = np.multiply(x, trem)
        I[:, :, c] = np.reshape(y, (pgc.height(I), pgc.width(I)))
    return I