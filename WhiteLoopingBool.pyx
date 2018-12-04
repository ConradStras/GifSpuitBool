import cv2
import cython
import numpy as np
from libcpp cimport bool
red = (0, 0, 255)

# generating the arrays that resemble a line through the image
# we cannot enter tuples into np.arange
# this code definitely requires optimizing
@cython.boundscheck(False)
@cython.cdivision(False)
cpdef whiteindex(unsigned char [:, :] image):
    cpdef int h, w, y, x
    cdef float  z = 0.02, tenth = 0.1, tff = 255
    cpdef int stepdown, stepacross
    cpdef double stepdownreal
    w = image.shape[1]
    h = image.shape[0]
    indexes = []
    stepdown = int(z * h)
    stepacross = int(z * w)
    for y in range(int(tenth * h), int(h - tenth * h), stepdown):
        for x in range(0, w, stepacross):
            if image[y, x] == tff:
                indexes.append((y, x))
    return indexes

def clusters(indexes, image):
    cpdef int w, zone1, zone2, ccount
    cpdef bint bzone
    w = image.shape[1]
    clustersindices = []
    h = image.shape[0]
    for j in indexes:
        ccount = 0
        y = j[0]
        x = j[1]
    # exception handling, i.e when a range lies within 6 of the edge of the image
        for k in range(y, y + int(h*1/100)-1):
            for i in range(x, x+int(w*1/100)-1):
                if image[k, i] == 255:
                    ccount = ccount + 1
        ##!! Key Parameter, this determines whether the amount of green present in a block is worthy of being a cluster
        threshold = int(1/45*(h*1/50*w*1/50))
        ##!! Key Parameter above
        if ccount > threshold:
            bzone = True
            clustersindices.append(((y, x), (y+int(h*1/100), x+int(w*1/100))))
    return clustersindices, bzone