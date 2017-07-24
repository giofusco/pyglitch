import numpy as np

def epoplectic(i, xshift, yshift):

    ydim = i.shape[0]
    xdim = i.shape[1]

    for x in range(xdim-11):
        for y in range(ydim-11):
            if np.sum(i[y,x]) > np.sum(i[y+yshift,x+xshift]):
                i[y,x] = i[y+10,x+10]
    return i