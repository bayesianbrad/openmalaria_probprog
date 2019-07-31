
import torch
import numpy as np
import sys
import math
from torch.distributions import Normal, Uniform
# import multiprocessing as mp
# from joblib import Parallel, delayed
import time
class f():
    def f(self):
        z1 = Normal(0,1).sample()
        z2 = self.R1(z1)
        # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
        # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
        while z2 == math.inf:
    #         print(' while loop triggered')
            z2 = self.R1(z1)
        z3 = Uniform(0,2).sample()
        z4 = self.R2(z2,z3)
        return torch.tensor([z1,z2,z3,z4])

    def R1(self,z1, i=0):
        temp = Normal(z1,1).sample()
        if temp.data > 0:
            return temp
        elif i == 1000:
            return math.inf
        else:
            i = i+1
            return self.R1(z1,i)

    def R2(self,z2,z3):
        ''' Need to ensure that we don't return a 
        value generated from a deterministic process, i.e if return temp=0, that
        has not been generated from the while loop - although with R1 defined as it is 
        this will never be the case'''
        temp = 0
        while temp < z2:
            temp = Normal(z3,1).sample()
        return temp
class frepeat():

    def __init__(self, z1,z3):
        self.z1 = z1
        self.z3 = z3
    def f(self):
        z2 = self.R1(self.z1)
        # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
        # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
        while z2 == math.inf:
            print(' while loop triggered')
            z2 = R1(self.z1)
        z4 = self.R2(z2,self.z3)
        return torch.tensor([self.z1,z2,self.z3,z4])

    def R1(self,z1, i=0):
        temp = Normal(z1,1).sample()
        if temp.data > 0:
            return temp
        elif i == 1000:
            return math.inf
        else:
            i = i+1
            return self.R1(z1,i)

    def R2(self,z2,z3):
        ''' Need to ensure that we don't return a 
        value generated from a deterministic process, i.e if return temp=0, that
        has not been generated from the while loop - although with R1 defined as it is 
        this will never be the case'''
        temp = 0
        while temp < z2:
            temp = Normal(z3,1).sample()
        return temp

class f1():
    def f(self):
        z1 = Normal(0,1).sample()
        z2 = self.R1(z1)
        # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
        # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
        while z2 == math.inf:
    #         print(' while loop triggered')
            z2 = R1(z1)
        z3 = Uniform(0,2).sample()
        z4 = self.R2(z2,z3)
        return torch.tensor([z1,z2,z3,z4])

    def R1(self,z1, i=0):
        temp = Normal(z1,1).sample()
        if temp.data > 0:
            return temp
        elif i == 1000:
            return math.inf
        else:
            i = i+1
            return self.R1(z1,i)

    def R2(self,z2,z3):
        ''' Need to ensure that we don't return a 
        value generated from a deterministic process, i.e if return temp=0, that
        has not been generated from the while loop - although with R1 defined as it is 
        this will never be the case'''
        temp = 0
        while temp < z2:
            temp = Normal(z3,1).sample()
        return temp



# class fvector():
#     def f(batch=1):
#         z1 = Normal(0,1).sample((batch,1))
#         z2 = R1(z1=z1, i=0, batch=batch)
#         # the following is used to avoid tail recursion  - simple solution and stops stack overflow 
#         # You can not use this trick straight forwardly if you wish to access multiple core / threads. 
#         while z2 == math.inf:
#     #         print(' while loop triggered')
#             z2 = R1(z1)
#         z3 = Uniform(0,2).sample()
#         z4 = R2(z2,z3)
#         return torch.tensor([z1,z2,z3,z4])

#     def R1(z1, i=0, batch=batch, trues, output):
#         """
#         batch  == orginal length of z1. 
#         """
#         if i>0:
#             if len(trues) >= 0:
#                 trues_prev = trues 
            
                
#         temp = Normal(z1,1).sample()
#         booleans = torch.gt(temp,torch.zeros(z1.shape))
#         if i == 0:
#             out_z1 = torch.zeros(z1.shape, dtype=torch.float)
#             nots = temp[~booleans]
#             trues = temp[booleans]
#         if len(nots) == 0:
#             return trues
#         elif i == 1000:
#             return math.inf
#         else:
#             i = i+1
#             if len(nots) >1:
#                 ''' get indicies and '''
#                 count = 0
#                 new_z1 = torch.zero(z1.shape(), dtype=torch.float)
#                 idx_nots = (booleans==0).nonzero()
#                 new_z1 = z1[idx_nots[:,0]] # would have to change this if the col of dim_col(z1) > 1

#             if len(trues) > 1:
#                 """ get indicies and store that, as we need to make sure the index output pairs are matched. """
#                 idx_trues = booleans.nonzero()
#                 for idx in idx_trues:
#                     out_z1[idx] =temp[idx]
#                 trues = torch.cat((trues_prev, trues),0) if len(trues) == 0 else trues_prev
#                 # new_z1 must preserve same shape and index characteristics as initial z1. 
#             return self.R1(z1=new_z1,i=i,trues=trues, output=out_z1)

    def R2(z2,z3):
        ''' Need to ensure that we don't return a 
        value generated from a deterministic process, i.e if return temp=0, that
        has not been generated from the while loop - although with R1 defined as it is 
        this will never be the case'''
        temp = 0
        while temp < z2:
            temp = Normal(z3,1).sample()
        return temp


