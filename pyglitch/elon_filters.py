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
#  Author: Elon Ulmann https://elonullman.com/

import numpy as np
import pyglitch.core as pgc


def epoplectic(I, xshift, yshift):
    ydim = pgc.height(I)
    xdim = pgc.width(I)

    xshift = int(xshift)
    yshift = int(yshift)

    assert xdim - xshift > 0 and xshift >= 0, "xshift must be between 0-%i" %xdim
    assert ydim - yshift > 0 and yshift >= 0, "yshift must be between 0-%i" %ydim

    for x in range(xdim - xshift):
        for y in range(ydim - yshift):
            if np.sum(I[y,x]) > np.sum(I[y + yshift, x+xshift]):
                I[y, x] = I[y + yshift, x+xshift]
    return I