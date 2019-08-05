from torch.utils.cpp_extension import load
import torch
# pytorch + c++

amortized_rs = load(name="amortized_rs",
                    sources=["amortized_rs.cpp"])
#
# for i in range(100):
#     a = amortized_rs.f()
#     print(a)

f_cpp = amortized_rs.f
import timeit
torch.manual_seed(7)
print(f_cpp())
print(timeit.timeit("f_cpp()", setup="from __main__ import f_cpp", number=2000))

# pure python
import math
import numpy as np
import torch
from torch.distributions import Normal, Uniform

def f():
    z1 = Normal(0,1).sample()
    ctr = 1
    z2 = R1(z1)
    # the following is used to avoid tail recursion  - simple solution and stops stack overflow
    # You can not use this trick straight forwardly if you wish to access multiple core / threads.
    while z2 == math.inf:
#         print(' while loop triggered')
        z2 = R1(z1)
        ctr += 1
    #print("CTR: ", ctr)
    z3 = Uniform(0,2).sample()
    #print(type(z3))
    z4 = R2(z2,z3)
    return torch.tensor([z1,z2,z3,z4])

def R1(z1, i=0):
    temp = Normal(z1,1).sample()
    if temp.data > 0:
        return temp
    elif i == 100:
        return math.inf
    else:
        i = i+1
        return R1(z1,i)

def R2(z2,z3):
    ''' Need to ensure that we don't return a
    value generated from a deterministic process, i.e if return temp=0, that
    has not been generated from the while loop - although with R1 defined as it is
    this will never be the case'''
    temp = 0
    while temp < z2:
        temp = Normal(z3,1).sample()
    return temp

import timeit
torch.manual_seed(7)
print(f())
print(timeit.timeit("f()", setup="from __main__ import f", number=2000))


# recursion-free
def R1(z1, i=0):
    i = 0
    bs = 1
    while True:
        #temp = Normal(z1, 1).sample((bs,))
        temp = Normal(z1, 1).sample()
        #mask = temp>0
        #if torch.sum(mask) > 0:
        #    return temp[mask][0]
        if temp > 0:
            return temp
        if i >= 10000:
            return math.inf
        i += bs

import timeit
torch.manual_seed(7)
print(f())
print(timeit.timeit("f()", setup="from __main__ import f", number=2000))