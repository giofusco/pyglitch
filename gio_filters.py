import pyglitch.audio_filtering as pgaf
import pyglitch.core as pgc


def swhoosh(I):
    I = pgaf.flanger(pgc.rotate_right(I), max_time_delay=.0055, rate=.0095, Fs=48100)
    I = pgc.rotate_left(I)
    I = pgaf.rescale_image(I)
    return I