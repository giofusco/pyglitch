import pyglitch.core as pgc
import pyglitch.filters as pgf
import pyglitch.audio_filtering as pgaf
import pyglitch.image_manipulation as pgim
import imageio as im
import numpy as np
import time

test_image = './images/sf.jpg'

print("\n \n \t ** pyGlitch v 0.1 -- Performance Tests **")
print("\t  \t written by Giovanni Fusco")

print("\n \n>> Loading ", test_image, "... ", end='')
I = pgc.open_image(test_image)
print("OK!", flush=True)


print("\n \n>> FLANGER [delay]: ", end='',  flush=True)
times = []
for d in np.arange(0.0, 0.1, 0.001):
    t = time.process_time()
    pgaf.flanger(I, max_time_delay=float(d), rate=0.005, Fs=48100, amp=0.9)
    times.append(time.process_time()-t)
    print("█", end='', flush=True)
print("\n ____ RESULTS ____ ")
print("Avg execution time:", np.mean(times[1:]), "s @",I.shape ,"[min: ", np.min(times[1:]), "s, max: ", np.max(times[1:]), "s]")


print("\n \n>> FLANGER  [rate]: ", end='',  flush=True)
times = []
for r in np.arange(0.0, 0.1, 0.005):
    t = time.process_time()
    pgaf.flanger(I, max_time_delay=0.01, rate=float(r), Fs=48100, amp=0.9)
    times.append(time.process_time() - t)
    print("█", end='', flush=True)
print("\n ____ RESULTS ____ ")
print("Avg execution time:", np.mean(times[1:]), "s @",I.shape ,"[min: ", np.min(times[1:]), "s, max: ", np.max(times[1:]), "s]")


print("\n \n>> TREMOLO  [Fc]: ", end='',  flush=True)
times = []
for fc in np.arange(0.0, 10., 0.5):
    t = time.process_time()
    pgaf.tremolo(I, Fc=fc, alpha=0.5, Fs=44100)
    times.append(time.process_time() - t)
    print("█", end='', flush=True)
print("\n ____ RESULTS ____ ")
print("Avg execution time:", np.mean(times[1:]), "s @",I.shape ,"[min: ", np.min(times[1:]), "s, max: ", np.max(times[1:]), "s]")


print("\n \n>> REVERB  [delay_pixels]: ", end='',  flush=True)
times = []
for d in range(5, I.shape[1], 5):
    t = time.process_time()
    pgaf.reverb(I, delay_pixels=d, decay = 0.5)
    times.append(time.process_time() - t)
    print("█", end='', flush=True)
print("\n ____ RESULTS ____ ")
print("Avg execution time:", np.mean(times[1:]), "s @",I.shape ,"[min: ", np.min(times[1:]), "s, max: ", np.max(times[1:]), "s]")


