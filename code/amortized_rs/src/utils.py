
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import math
import time
import os
import sys
import warnings
import torch.multiprocessing as mp
from torch.multiprocessing import cpu_count
import random
from time import strftime
from rejection_samplers import frepeat
from joblib import Parallel, delayed
from tqdm import tqdm

def load_samples(samples_per_file=95000, saved_entries=4, PATH=None, learning=True):
    """
     This function assumes that the samples must be concatenated and are saved in a folder specified by
     PATH. If you want the data back in learning form i.e training, validation and test specify `learning = True`.

     param: samples_per_file :type: int :descrip: How many samples are contained in each file
     param: saved_entries :type: int :descrip: How many latten variables are saved
     param: PATH type: string :descrip: Path where samples are saved 
     param: learning type: bool descrip: True splits training into training, validation and test and returns 3 tensors, else, one tensor of all samples. 
    """
    count = 0
    with os.scandir(PATH) as files:
        for file in files:
            temp = torch.load(os.path.join(PATH,file.name))
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
        test_samples = samples[nvalidation_samples:total_samples-1,:]
        return train_samples, validation_samples, test_samples
    
    else:
        return samples

def gen_samples_parallel(total_samples=1000, nOutputs=4, simulator=None, UNIQUE_ID=None, Save_PATH=None, nCheckpoint=10):
    """
    A function to generate samples for different simulators loops.

    :param: total_samples :type: int :descrip: How many samples to generate in total
    :param: n_outputs :type: int :descrip: Defines the number of outputs, outputted by the simulator.
    :param: simulator :type: class :descrip: Takes an instantiation of a class that defines a simulator. Ensure the name 
            of the main simulator function in the class is called `<class_name>.f()`. 
    
    :return saves samples to directory in which code is called
    TODO: add Path argument later on.
    """
    if not UNIQUE_ID:
        UNIQUE_ID = str(random.randint(0,1000000)) + '_' + strftime("%Y-%m-%d_%H-%M")
    
    samples = torch.zeros(total_samples,nOutputs)
    nCpus = cpu_count()  
    start = time.time()
    sim = simulator()
    def run(sim):
        return sim.f()

    results = Parallel(n_jobs=nCpus)(delayed(run)(sim) for i in tqdm(range(total_samples)))
    
    print('samples collected ')
    k = 0
    for result in results:
        samples[k,:] = result
        k += 1
    
    end = time.time()
    total_time = end - start
    print(' Time taken is {} for {} samples for UNIQUE_ID: {}'.format(total_time, total_samples,UNIQUE_ID))
    return samples

def gen_samples_non_parallel(total_samples=1000, n_outputs=4, simulator=None, PATH='results/', UNIQUE_ID=None):
    """
    A function to generate samples for different simulators loops. 
    Due to the infinite expected computation time of some while loops, it is highly 
    probably that we will trigger a stack overflow attack. 

    :param: total_samples :type: int :descrip: How many samples to generate in total
    :param: n_outputs :type: int :descrip: Defines the number of outputs, outputted by the simulator.
    :param: function :type: class :descrip: Takes an instantiation of a class that defines a simulator. Ensure the name 
            of the main simulator function in the class is called `<class_name>.f()`. 
    
    :return saves samples to a folder in the directory in which code is called, called results
    """
    samples = torch.zeros(total_samples,n_outputs)
    if not UNIQUE_ID:
        UNIQUE_ID = str(random.randint(0,1000000)) + '_' + strftime("%Y-%m-%d_%H-%M")
    if simulator:
        sim = simulator()
    start = time.time()
    for i in range(total_samples):
        if i % 5000 == 0 and i != 0:
            print('{} samples'.format(i))
            torch.save(samples, 'Gensamples_{}_{}.pt'.format(i,UNIQUE_ID))
        if i == total_samples - 1:
            print('{} samples'.format(i))
            torch.save(samples, 'Gensamples_{}_{}.pt'.format(i,UNIQUE_ID))           
        samples[i,:] = sim.f()

    end = time.time()
    total_time = end - start
    print(' Time taken is {} for {} samples'.format(total_time, total_samples))

def create_dataset(data=None, PATH=None, batch_size=5, totalSamples=1000, nOutputs=4, indx_latents=None, simulator=None, UNIQUE_ID=None, Save_PATH='results/'):
    """
    For each datum in your data set, this function loads the simulator and 
    creates a dataset from the rejection samplers corresponding to batch size. 

    :param: data :type: torch.tensor :descrip: Can either pass the tensor directly through, or PATH. 
    :param: PATH :type: str :descrip: Path to data, in the form of torch tensor. 
    :param: batch_size :type: int :descrip: How many copies of the original input to run through the simulator. 
    ;param indx_latents :type: list :descrip: A list of ints of which column the latent variables are stored in. 

    :return: dataset :type: torch.tensor :descrip: for batch training. 
    """

    # if data:
    #     data = data

    if PATH:
        data = torch.load(PATH)
    
    if not UNIQUE_ID:
        UNIQUE_ID = str(random.randint(0,1000000)) + '_' + strftime("%Y-%m-%d_%H-%M")
    
    if not os.path.exists(Save_PATH):
        os.makedirs(Save_PATH)
    

    nCpus = cpu_count()
    genSamples = torch.zeros(totalSamples*batch_size,nOutputs)
    origSamples = data
    def run(i,batch_size, origSamples, simulator):
        newSamples=  torch.zeros(batch_size, nOutputs)
        sim = simulator(origSamples[i,0],origSamples[i,2])
        for j in range(batch_size):
            newSamples[j,:] = sim.f()
        torch.save('batch_samples_{}.pt'.format(i),newSamples)
        return newSamples
    
    start = time.time()
    results = Parallel(n_jobs=nCpus)(delayed(run)(i, batch_size,origSamples,simulator) for i in tqdm(range(totalSamples)))
    print(' Batched samples collected')
    print('='*50)
    k = 0
    l = 0
    for result in results:
        genSamples[l:batch_size+k*batch_size,:] = result
        k += 1
        l += batch_size 
    
    end = time.time()
    totalTime = end - start

    samplePath =  Save_PATH + 'BatchSamples_{}_{}.pt'.format(UNIQUE_ID)
    torch.save(genSamples, samplePath)
    print(' Time taken is {} for {} samples'.format(totalTime, totalSamples))
    print('='*50)
    print(' Batched Samples saved to {}'.format(samplePath))




        
class RejectionDataset(Dataset):
    def __init__(self, split, l_data, train_percentage, fname_test, fname_train, InIndx, OutIndx):
        
        _samples = torch.load('/data/'+fname_test)
        n_split = (len(_samples)*train_percentage//l_data)*l_data
        
            
        self.zinput = _samples[:n_split,InIndx] if split == 'train' else torch.load("data/"+fname_test)[n_split:]
        self.zoutput = samples[:n_split,OutIndx] if split == 'train' else torch.load("data/"+fname_test)[n_split:]
        # using shuffle= True does the same job
        # perm = torch.randperm(len(self.zinput))
        # self.zinput, self.zoutput = self.zinput[perm], self.zoutput[perm]
        # self.l_data = l_data

    def __getitem__(self, idx):
        ' Generates one sample of data - in our case batch size'
        start_idx = idx*self.l_data
        end_idx = start_idx + self.l_data
        return self.zinput[start_idx:end_idx], self.zoutput[start_idx:end_idx]
    
    # def __len__(self):
    #     ' Denotes the total number of samples'
    #     return len(self.list_IDs)
    
