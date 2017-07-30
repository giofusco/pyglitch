import numpy as np
import pyglitch.core as pgc


def epoplectic(i, xshift, yshift):
    ydim = pgc.height(i)
    xdim = pgc.width(i)

    assert xdim - xshift > 0 and xshift > 0, "xshift must be between 0-%i" %xdim
    assert ydim - yshift > 0 and yshift > 0, "yshift must be between 0-%i" %ydim

    for x in range(xdim - xshift):
        for y in range(ydim - yshift):
            if np.sum(i[y,x]) > np.sum(i[y + yshift, x+xshift]):
                i[y, x] = i[y + yshift, x+xshift]
    return i