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
batch_f = amortized_rs.batch_f
import timeit
torch.manual_seed(7)
print(f_cpp())
torch.manual_seed(7)
print(batch_f(1))
torch.manual_seed(7)
print(timeit.timeit("f_cpp()", setup="from __main__ import f_cpp", number=2000))
torch.manual_seed(7)
print(timeit.timeit("[f_cpp() for _ in range(2000)]", setup="from __main__ import f_cpp", number=1))
torch.manual_seed(7)
print(timeit.timeit("batch_f(2000)", setup="from __main__ import batch_f", number=1))

# pure python
import math
import numpy as np
import torch
from torch.distributions import Normal, Uniform

def s():
    ''' In this simple simulator we care about the input output sets:
    {z1, z2}
    {(z2,z3), z4}
    {z5
    '''
    z1 = Normal(0,1).sample()
    z2 = R1(z1)
    # the following is used to avoid tail recursion  - simple solution and stops stack overflow
    # You can not use this trick straight forwardly if you wish to access multiple core / threads.
    while z2 == math.inf:
        # we resample again to increase chance of acceptance.
        # This does not change acceptance probabilities. This is just for
        # computational efficiency we generating samples.
        z1 = Normal(0, 1).sample()
        z2 = R1(z1)
    z3 = Uniform(0,2).sample()
    z4 = R2(z2,z3)
    z5 = Normal(50,30).sample()
    M = torch.tensor(3.0337) # determined via x = torch.linsapce(-50.0,151.0, nsamples); torch.max(f(x)/g(x))
    z6, gz, fz = R3(z5,M)
    while z6 == math.inf:
        # we resample again to increase chance of acceptance.
        # This does not change acceptance probabilities. This is just for
        # computational efficiency we generating samples.
        z5 = Normal(50,30).sample()
        z6 = R3(z5, M)
    return torch.tensor([z1,z2,z3,z4,z5,z6])

def f(x, mu1, sigma1, mu2, sigma2):
    ' PDF of Mixture of gaussian that we want to sample from: N(mu1,sigma1) + N(mu2,sigma2)'
    const1 =  1 / (2 * np.pi * sigma1 ** 2 *torch.ones(x.shape)) ** 0.5
    const2 =  1 / (2 * np.pi * sigma2 ** 2 *torch.ones(x.shape)) ** 0.5
    body1 =  torch.exp(-(x - mu1*torch.ones(x.shape)) ** 2 / (2 * sigma1 ** 2 *torch.ones(x.shape)))
    body2 =  torch.exp(-(x - mu2*torch.ones(x.shape)) ** 2 / (2 * sigma2 ** 2 *torch.ones(x.shape)))
    return  const1*body1 + const2*body2

def g(x,mu1, sigma1):
    ' PDF of simple normal proposal N(mu1, sigma1)'
    const = 1 / (2 * np.pi * sigma1 ** 2 *torch.ones(x.shape)) ** 0.5
    body = torch.exp(-(x - mu1) *torch.ones(x.shape) ** 2 / (2 * sigma1 ** 2 *torch.ones(x.shape)))
    return const * body

def R3(z,M):
    ' This fucntion will always return z back, eventually'
    tempg = g(z, 50,30)
    tempf = f(z,30,10,80,20)
    u = Uniform(0,M*tempg).sample()
    i = 0
    bs = 1
    while True:
        if u <= tempf:
            return z
        if i>=10000:
            return math.inf
        i += bs


def R2(z2,z3):
    ''' Need to ensure that we don't return a
    value generated from a deterministic process, i.e if return temp=0, that
    has not been generated from the while loop - although with R1 defined as it is
    this will never be the case'''
    temp = 0
    while temp < z2:
        temp = Normal(z3,1).sample()
    return temp

# recursion-free
def R1(z1, i=0):
    i = 0
    bs = 1
    while True:
        temp = Normal(z1, 1).sample()
        if temp > 0:
            return temp
        if i >= 10000:
            return math.inf
        i += bs


def f(x, mu1, sigma1, mu2, sigma2):
    return 1 / (2*np.pi*sigma1**2)**0.5 * torch.exp(-(x-mu1)**2 / 2*sigma1**2)  + 1 / (2*np.pi*sigma2**2)**0.5 * torch.exp(-(x-mu2)**2 / 2*sigma2**2)

def g(x,mu1, sigma1):
    return 1 / (2 * np.pi * sigma1 ** 2) ** 0.5 * torch.exp(-(x - mu1) ** 2 / 2 * sigma1 ** 2)

def R3(z5):
    k = max(f(z5,30,10,80,20)/g(z5,50,30))
    u = Uniform(0, k*g(z5, 50,30)).item()
    x = True
    while x:
        if u <= f(z5, 30, 10, 80, 20).item():
            x = False
        else:
            continue
    return z5


# def plotR3(x)
#     plt.plot(x, f(x))
#     plt.plot(x, k*g(x))
#     plt.show()
#
#     s = R3(iter=100000)
#     sns.distplot(s)

import timeit
torch.manual_seed(7)
print(s())
print(timeit.timeit("f()", setup="from __main__ import f", number=2000))

amortized_rs_std = load(name="amortized_rs_std",
                    sources=["amortized_rs_std.cpp"])

f_cpp = amortized_rs_std.f
batch_f = amortized_rs_std.batch_f
import timeit
torch.manual_seed(7)
print(f_cpp())
print(timeit.timeit("f_cpp()", setup="from __main__ import f_cpp", number=2000))
print(timeit.timeit("batch_f(2000)", setup="from __main__ import batch_f", number=1))