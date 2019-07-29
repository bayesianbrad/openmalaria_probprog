
import torch
import numpy as np
import sys
import math
import time
import os
import sys




def load_samples(samples_per_file=95000, saved_entries=4, PATH="/samples_with_inputs/", learning=True):
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

def gen_samples(total_samples=1000, n_outputs=4, simulator=sim):
    """
    A function to generate samples for different simulators loops. 

    :param: total_samples :type: int :descrip: How many samples to generate in total
    :param: n_outputs :type: int :descrip: Defines the number of outputs, outputted by the simulator.
    :param: function :type: class :descrip: Takes an instantiation of a class that defines a simulator. Ensure the name 
            of the main simulator function in the class is called `<class_name>.f()`. 
    
    :return saves samples to directory in which code is called
    TODO: add Path argument later on.
    """
    n_samples = int(total_samples/10)
    samples_1 = torch.zeros(n_samples,n_outputs)
    samples_2 = torch.zeros(n_samples,n_outputs)
    samples_3 =torch.zeros(n_samples,n_outputs)
    samples_4 = torch.zeros(n_samples,n_outputs)
    samples_5 = torch.zeros(n_samples,n_outputs)
    samples_6 = torch.zeros(n_samples,n_outputs)
    samples_7 = torch.zeros(n_samples,n_outputs)
    samples_8 = torch.zeros(n_samples,n_outputs)
    samples_9 = torch.zeros(n_samples,n_outputs)
    samples_10 = torch.zeros(n_samples,n_outputs)

    # # could write as list comprehension
    # num_cores = mp.cpu_count()

    # results = Parallel(n_jobs=num_cores)(delayed(f)() for i in range(n_samples))
    # print(results)
    # index = [i for i in range(1,11)]
    start = time.time()
    for i in range(n_samples):
        if i % 5000 == 0 and i != 0:
            print('{} samples'.format(i))
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
                
        samples_1[i,:] = sim.f()
        samples_2[i,:] = sim.f()
        samples_3[i,:] = sim.f()
        samples_4[i,:] = sim.f()
        samples_5[i,:] = sim.f()
        samples_6[i,:] = sim.f()
        samples_7[i,:] = sim.f()
        samples_8[i,:] = sim.f()
        samples_9[i,:] = sim.f()
        samples_10[i,:] = sim.f()

    end = time.time()
    total_time = end - start
    print(' Time taken is {} for {} samples'.format(total_time, n_samples))
