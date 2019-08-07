import numpy as np
cimport numpy as cnp

def  f():
    DTYPE = cnp.float64_t
    retarr = np.zeros((4,), dtype=DTYPE)
    cdef double z_1 = cnp.random.normal(0, 1, 1)
    cdef double z_2
    while z_2 == cnp.inf:
        z_2 = R1(z_1, 1)
    cdef double z_3 = cnp.random.uniform(0, 2)
    cdef double z_4 = R2(z_2, z_3, 1)
    retarr[0] = z_1
    retarr[1] = z_2
    retarr[2] = z_3
    retarr[3] = z_4
    return retarr

cdef double R1(double z_1, int i) nogil:
    cdef double temp = cnp.random.normal(z_1, 1, 1)
    if temp > 0:
        return temp
    elif i == 1000:
        return cnp.inf
    else:
        i += 1
        return R1(z_1,i)

cdef double R2(double z_2, double z_3) nogil:
    cdef double temp = 0
    while temp < z_2:
        temp = cnp.random.normal(z_3, 1, 1)
    return temp

# cdef f_nonrec():
