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