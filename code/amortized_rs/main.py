import torch
import numpy as np
from torch.distributions import Normal, Uniform
import sys
import math
import multiprocessing as mp
from joblib import Parallel, delayed
import time
import os
import sys

class f():
    def f():
        z1 = Normal(0,1).sample()
        z2 = R1(z1)
        # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
        # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
        while z2 == math.inf:
    #         print(' while loop triggered')
            z2 = R1(z1)
        z3 = Uniform(0,2).sample()
        z4 = R2(z2,z3)
        return torch.tensor([z1,z2,z3,z4])

    def R1(z1, i=0):
        temp = Normal(z1,1).sample()
        if temp.data > 0:
            return temp
        elif i == 1000:
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

class f1():
    def f():
        z1 = Normal(0,1).sample()
        z2 = R1(z1)
        # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
        # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
        while z2 == math.inf:
    #         print(' while loop triggered')
            z2 = R1(z1)
        z3 = Uniform(0,2).sample()
        z4 = R2(z2,z3)
        return torch.tensor([z1,z2,z3,z4])

    def R1(z1, i=0):
        temp = Normal(z1,1).sample()
        if temp.data > 0:
            return temp
        elif i == 1000:
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



class fvector():
    def f(batch=1):
        z1 = Normal(0,1).sample((batch,1))
        z2 = R1(z1)
        # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
        # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
        while z2 == math.inf:
    #         print(' while loop triggered')
            z2 = R1(z1)
        z3 = Uniform(0,2).sample()
        z4 = R2(z2,z3)
        return torch.tensor([z1,z2,z3,z4])

    def R1(z1, i=0,batch=5):
        temp = Normal(z1,1).sample((batch,1)
        booleans = torch.gt(temp,torch.zeros(z1.shape))
        nots = z1[~booleans]
        trues = z1[booleans]

            return temp
        elif i == 1000:
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

# # could write as list comprehension
# num_cores = mp.cpu_count()

# results = Parallel(n_jobs=num_cores)(delayed(f)() for i in range(n_samples))
# print(results)
def gen_samples(n_samples=1000):
    samples_1 = torch.zeros(n_samples,4)
    samples_2 = torch.zeros(n_samples,4)
    samples_3 =torch.zeros(n_samples,4)
    samples_4 = torch.zeros(n_samples,4)
    samples_5 = torch.zeros(n_samples,4)
    samples_6 = torch.zeros(n_samples,4)
    samples_7 = torch.zeros(n_samples,4)
    samples_8 = torch.zeros(n_samples,4)
    samples_9 = torch.zeros(n_samples,4)
    samples_10 = torch.zeros(n_samples,4)

    # # could write as list comprehension
    # num_cores = mp.cpu_count()

    # results = Parallel(n_jobs=num_cores)(delayed(f)() for i in range(n_samples))
    # print(results)
    # index = [i for i in range(1,11)]
    start = time.time()
    for i in range(n_samples):
        if i % 5000 == 0:
            print('{} samples'.format(i))
            if i == 0:
                continue
            else:
                torch.save(samples_1, 'samples_1_{}.pt'.format(i))
                torch.save(samples_2, 'samples_2_{}.pt'.format(i))
                torch.save(samples_3, 'samples_3_{}.pt'.format(i))
                torch.save(samples_4, 'samples_4_{}.pt'.format(i))
                torch.save(samples_5,'samples_5_{}.pt'.format(i))
                torch.save(samples_6, 'samples_6_{}.pt'.format(i))
                torch.save(samples_7, 'samples_7_{}.pt'.format(i))
                torch.save(samples_8, 'samples_8_{}.pt'.format(i))
                torch.save(samples_9, 'samples_9_{}.pt'.format(i))
                torch.save(samples_10,'samples_10_{}.pt'.format(i))
                
        samples_1[i,:] = f()
        samples_2[i,:] = f()
        samples_3[i,:] = f()
        samples_4[i,:] = f()
        samples_5[i,:] = f()
        samples_6[i,:] = f()
        samples_7[i,:] = f()
        samples_8[i,:] = f()
        samples_9[i,:] = f()
        samples_10[i,:] = f()

    end = time.time()
    total_time = end - start
    print(' Time taken is {} for {} samples'.format(total_time, n_samples))


def load_samples(samples_per_file=95000, saved_entries=4, PATH="/samples_with_inputs/", learning=True):
    """
     param: tot_samples
     param: saved_entries descrip: How many latten variables are saved
    """
    count = 0
    with os.scandir(PATH) as files:
        for file in files:
            temp = torch.load(PATH + file.name)
            # indexing starts at 1, because the first entry of all samples is 0. 
            if count == 0:
                samples = temp[1:samples_per_file,:]
            else:
                samples = torch.cat((samples, temp[1:samples_per_file,:]),0)
            count += 1
    
    if learning:
        # simple split for training, validation and test data. 
        total_samples = len(samples)
        ntrain_samples = int(0.7*total_samples)
        nvalidation_samples = int(0.8*total_samples)
        train_samples = samples[:ntrain_samples,:]
        validation_samples = samples[ntrain_samples:nvalidation_samples,:]
        test_samples = samples[nvalidation_samples:end,:]
        return train_samples, validation_samples, test_samples
    
    else:
        return samples


class density_estimator(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


        
    