import pyglitch.core as core
import numpy as np
import math

def reverb(I, patch, delay_pixels, decay = 0.5, convert_to_one_dimensional=False):

    if convert_to_one_dimensional:
        _patch = np.array(patch[4]).ravel()
        patch = list(patch)
        for i in range(0, len(_patch) - delay_pixels-1):
            # WARNING: overflow potential
            _patch[i + delay_pixels] += (_patch[i] * decay).astype(np.uint8)
        patch[4] = np.reshape(_patch, patch[4].shape)
    else:
        for i in range(0, patch[2] - delay_pixels-1):
        # WARNING: overflow potential
            patch[4][:-1, i + delay_pixels] += (patch[4][:-1, i] * decay).astype(np.uint8)
    core.put_patch_in_place(I, patch)
    return I

def wah_wah(I, damp=0.05, minf=500, maxf=5000, Fw=2000, Fs=44100):

    delta = Fw / Fs
    x = np.array(I).ravel()
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
    for n in range(2, len(x)-1):
        yh[n] = x[n] - yl[n - 1] - Q1 * yb[n - 1]
        yb[n] = F1 * yh[n] + yb[n - 1]
        yl[n] = F1 * yb[n] + yl[n - 1]
        F1 = 2 * math.sin((math.pi * Fc[n]) / Fs)
    maxyb = max(abs(yb))
    yb = yb / maxyb
    I = np.reshape(yb, I.shape)
    return I