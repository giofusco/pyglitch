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

import numpy as np
import pyglitch.core as pgc
import pyglitch.image_manipulation as pgim
import pyglitch.audio_filtering as pgaf


def vhs(X, blur_amount, shift_amount, shift_channel):
    I = X.copy()
    H = np.ones((1,blur_amount))
    H = np.asarray(H)*1/blur_amount
    I = pgim.shift_channel_hor(I, shift_channel, shift_amount)
    I = pgim.apply_filter(I, H)
    I = pgaf.flanger(I, max_time_delay=0.0001, rate=.75, Fs=48100)
    return I


def grid(X, thickness, darkness):
    I = X.copy()
    # I = pgaf.tremolo(I, Fc=1.5, alpha=.25, Fs=48100)
    I = pgaf.tremolo(I, Fc=thickness, alpha=darkness, Fs=48100)
    I = pgaf.tremolo(pgc.rotate_right(I), Fc=thickness, alpha=darkness, Fs=48100)
    I = pgc.rotate_left(I)
    return I