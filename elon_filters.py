import numpy as np
import pyglitch.core as pgc

def epoplectic(i, xshift, yshift):

    ydim = pgc.height(i)
    xdim = pgc.width(i)

    # TODO: check range of indexes when accessing matrix i
    #       why -11 in range?
    #
    for x in range(xdim-11):
        for y in range(ydim-11):
            if np.sum(i[y,x]) > np.sum(i[y+yshift,x+xshift]):
                i[y,x] = i[y+10,x+10]
    return i