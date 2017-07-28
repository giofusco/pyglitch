import pyglitch.core as pgc
import numpy as np
import math

def reverb(I, delay_pixels, decay = 0.5):
    assert delay_pixels > 0, "delay_pixels must be positive integer"
    x = pgc.to_1d_array(I)
    for i in range(0, len(x) - delay_pixels-1):
        # WARNING: overflow potential
        # x[i + delay_pixels] += (x[i] * decay).astype(np.uint8)
        x[i + delay_pixels] += (x[i] * decay)
    I = np.reshape(x, I.shape)
    return I.astype(np.uint8)


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
    for n in range(2, len(x)-1):
        yh[n] = x[n] - yl[n - 1] - Q1 * yb[n - 1]
        yb[n] = F1 * yh[n] + yb[n - 1]
        yl[n] = F1 * yb[n] + yl[n - 1]
        F1 = 2 * math.sin((math.pi * Fc[n]) / Fs)
    maxyb = max(abs(yb))
    yb = yb / maxyb
    I = np.reshape(yb, I.shape)
    return I


def flanger(I, max_time_delay = 0.003, rate=1, Fs=44100):
    x = pgc.to_1d_array(I)
    idx = np.arange(0, len(x))
    sin_ref = (np.sin(2 * math.pi * idx * (rate / Fs)))
    max_samp_delay = round(max_time_delay * Fs)
    y = np.zeros(len(x))
    y[1: max_samp_delay] = x[1: max_samp_delay]
    amp = 0.7
    for i in range(max_samp_delay+1, len(x)):
        cur_sin = np.abs(sin_ref[i])
        cur_delay = math.ceil(cur_sin * max_samp_delay)
        y[i] = (amp * x[i]) + amp * (x[i - cur_delay])
    I = np.reshape(y, I.shape)
    return I.astype(np.uint8)


def tremolo(I, Fc=5, alpha=0.5, Fs=44100):
    x = pgc.to_1d_array(I)
    index = np.arange(0, len(x))
    trem = (1 + alpha * np.sin(2 * np.pi * index * (Fc / Fs)))
    y = np.multiply(x,trem)
    I = np.reshape(y, I.shape)
    return I.astype(np.uint8)
