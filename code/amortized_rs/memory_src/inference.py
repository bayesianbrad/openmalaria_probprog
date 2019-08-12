import torch.multiprocessing as mp
from model import density_estimator
from torch.utils.cpp_extension import load
import torch as th
from torch import optim
import torch.distributions as dist
from time import strftime
import os
import time
import shutil
import logging

amortized_rs = load(name="amortized_rs",
                    sources=["amortized_rs.cpp"])


class Inference():
    """
    Class for training and testing the regressor.
    """

    def __init__(self, address,
                       batchSize,
                       model,
                       optimizer,
                       lossFn,
                       nIterations,
                       device='cpu',
                       loadData=False,
                       loadDatafName=None,
                       logPath=None,
                       loadCheckpoint=None):
        '''

        :param address: :type str Which rejection sampling blocks to amortize
        :param batchSize: :type int Number of samples to push at each iteration.
        :param model: :type class Neural network architecture 
        :param optimizer: :type torch.optim Pytorch optimizer
        :param lossFn: :type torch.loss Pytorch loss function
        :param nIterations: :type int Number of iterations for training
        :param device: :type str Device to be used for data generation
        :param loadData: :type bool determines whether to upload pre-loaded data (true), or generate data on the fly (False)
        :param loadDatafName :type str load file name
        :param logPath :type str log save path
        :param loadCheckpoint :type File to load to resume from checkpoint.
        '''
        self.address = address
        self.batchSize = batchSize
        self.model = model
        self.optimizer = optimizer
        self.lossFn = lossFn
        self.nIterations = nIterations
        self.device = device
        self.loadData = loadData
        self.loadDatafName = loadDatafName
        self.logPath = logPath
        if loadCheckpoint:
           logging.info("Restoring parameters from {}".format('../checkpoint/'+loadCheckpoint))
           self.load_checkpoint(loadCheckpoint)

        self.inputSize = inputSize
        self.outpuSize = outputSize
        self.processes = mp.cpu_count()


    # no longer need write the batch loop in c - we can do that later
    def get_batch(self, count):
        '''
        This function gathers one batch of data, either from a pre-generated data source
        or generates the data on the fly, if there is no pre-generated data.
        In the future, this function is to be replaced with the data generated from the
        pyprob_cppp protocols.

        :param count: :type int When loading data this tracks which indices to load 
        :return: inR* :type torch.tensor The input data to the rejection sampling block
                outR* :type torch.tensor The output data from the rejection sampling block. 
                count :type int Updated count
        '''
        if not self.self.loadData:
            data = th.stack([amortized_rs.f() for _ in range(batch_size)])
        if self.address == 'R1' and not self.loadData:
            inR1 = data[0:batch_size, 0].view([batch_size, 1]).view([1, batch_size])
            outR1 = data[0:batch_size, 1].view([batch_size, 1]).view([1, batch_size])
            count = count + 1  # not actually need if self.loadData == True
            return inR1.to(device), outR1.to(self.device), count
        elif self.address == 'R1' and self.loadData:
            inR1 = data[count:batch_size + count * batch_size, 0]
            outR1 = data[count:batch_size + count * batch_size, 1]
            count = count + 1
            return inR1.to(self.device), outR1.to(self.device), count
        elif self.address == 'R2' and not self.loadData:
            inR2 = th.cat([data[0:batch_size, 1], data[0:batch_size, 2]], dim=0).view(1, 256)
            outR2 = data[0:batch_size, 3].view([1, batch_size])
            count = count + 1
            return inR2.to(self.device), outR2.to(self.device), count
        elif self.address == 'R2' and self.loadData:
            inR2 = th.stack(
                [data[count:batch_size + count * batch_size, 1], data[count:batch_size + count * batch_size, 2]], dim=1)
            outR2 = data[count:batch_size + count * batch_size, 1]
            count = count + 1
            return inR2.to(self.device), outR2.to(self.device), count


    def train(self, process, saveModel=True, saveName=None, checkpoint=True):
        '''
        Train model

        :param process: :type int Process number, when used in parallel.
        :param saveModel :type bool To save model set True.
        :param saveName :type str Default saveName for model is a time stamp + process name.
        :param checkpoint :type bool If you want to store avg loss data, for each iteration, model weights etc
        :return:
        '''
        self.model.train()


        if self.loadData == True:
            nSamples = self.batchSize * self.nIterations
            data = th.load('../data/all_batch_samples.pt')
            data = data[0:nSamples, :]

        self.optimizer.zero_grad()
        count = 0
        _outLoss = 0

        # The network needs to learn z1 -> z2 and z2,z3 -> z4
        for i in range(self.nIterations):
            inData, outData, count = self.get_batch(count)
            proposal = dist.Normal(*self.model(inData))
            pred = proposal.rsample(sample_shape=[self.batchSize]).view(1, 128)
            self.optimizer.zero_grad()
            _loss = self.lossFn(outData, pred)
            _loss.backward()
            self.optimizer.step()

            _outLoss += _loss.item()
            # if _outLoss == nan:
            #     break
            if i % 100 == 0:
                avgLoss = _outLoss / (i + 1)
                print("iteration {}: Average loss: {}".format(i, avgLoss))
                if checkpoint:
                    checkpointData = {'model': density_estimator(),
                                      'state_dict': self.model.state_dict(),
                                      'iteration': i,
                                      'optimizer': self.optimizer.state_dict(),
                                      'addresss': self.address,
                                      'avg_loss': avgLoss
                                      }
                    self.save_checkpoint(checkpointData)

        if not os.path.exists('../model/'):
            os.makedirs('../model/')
        if saveModel:
            if saveName:
                fname = '../model/{}_process_{}_address_{}'.format(saveName, process, self.address)
            else:
                fname = '../model/model_{}_process_{}_address_{}'.format(strftime("%Y-%m-%d_%H-%M"), process, self.address)
            th.save(self.model.state_dict(), fname)
            print(' Model is saved at : {}'.format(fname))

    def save_checkpoint(self, state):
       """
       Saves model and training parameters at checkpoint + 'last.pth.tar'.
       :param: state :type dict Contains model's state, may contain other keys such as epoch, optimizer

       """
       if not os.path.exists('../checkpoints/'):
           os.mkdir('../checkpoints/')
       timeStamp = strftime("%Y-%m-%d_%H-%M")
       fName = '../checkpoints/' + os.path.join(timeStamp, 'last.pth.tar')
       th.save(state, fName)

    def load_checkpoint(self, checkpoint):
        """Loads model parameters (state_dict) from file_path.

        :param: checkpoint: :type str Filename which needs to be loaded
        :param: model: :type torch.nn.Module  Model for which the parameters are loaded
        :param: optimizer: :type torch.optim  Optional: resume optimizer from checkpoint
        """
        # assumes only file name is passed.
        checkpoint = '../checkpoints/'+checkpoint+'last.pth.tar'
        if not os.path.exists(checkpoint):
            raise ("File doesn't exist {}".format(checkpoint))
        checkpointData = th.load(checkpoint)
        self.model = self.model.load_state_dict(checkpointData['state_dict'])
        self.optimizer = self.optimizer.load_state_dict(checkpointData['optimizer'])
        self.address = checkpointData['address']

        return checkpointData

    def test(self, testIterations, modelName):
        '''

        :param testIterations:
        :param modelName:
        :return:
        '''
        modelName = '../model/' + modelName
        self.model.load_state_dict(th.load(modelName))
        self.odel.eval()
        _outloss = 0
        count = 0
        with th.no_grad():
            for i in range(testIterations):
                inData, outData, count = self.get_batch(self.address, self.batchSize, count)
                proposal = dist.Normal(*model(inData))
                pred = proposal.rsample(sample_shape=[self.batchSize]).view(1, 128)
                # learns a \hat{y} for the whole batch and then generates n_batch_size samples to predict the output.
                _loss = self.loss_fn(outData, pred)
                _outloss += _loss.item()
                if i % 100 == 0:
                    print('{} Iteration , Test avg loss: {}\n'.format(i + 1, _outloss / (i + 1)))
            print('Test loss: {}\n'.format(_outloss / testIterations))


    def objective(zlearn, *args, **kwargs):
        ''' This has to be representative of the objective, equation 2'''
        return 0

    def run(self, *args, **kwargs):
        '''
        Runs the methods
        :param args:
        :param kwargs:
        :return:
        '''

        if self.trainOn:
            model.share_memory()
            processes = []
            keywords =  {'saveModel':kwargs['saveModel'], 'saveName':kwargs['saveName'],
                         'checkpoint':kwargs['checkpoint'], 'loadCheckpoint':kwargs['loadCheckpoint']}
            for process in range(self.processes):
                p = mp.Process(target=self.train, args=process, kwargs=keywords)
                p.start()
                processes.append(p)
            for p in self.processes:
                p.join()
        testOn = False
        if testOn:
            model_name = 'model_2019-08-08_19-15_rejectionBlock_R1_process_2'
            n_test = 1000
            self.test(self.testIterations, self.modelName)


if __name__ == '__main__':
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device("cpu")
    lr = 0.001
    momentum = 0.9
    self.loadData = False
    batch_size = 128
    self.address = 'R2'
    outputSize = 1
    # outputSize = 128 # for R1 and R2
    if self.address == 'R2':
        inputSize = batch_size * 2
    if self.address == 'R1':
        inputSize = batch_size
    model = density_estimator(inputSize, outputSize)
    num_processes = mp.cpu_count()
    N = 2000
    trainOn = True
    loss_fn = th.nn.MSELoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # train(model, optimizer, loss_fn, N, self.address, batch_size, rank=0, self.loadData=False)
    # NOTE: this is required for the ``fork`` method to work
    if trainOn:
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model, optimizer, loss_fn, N, self.address, batch_size, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    testOn = False
    if testOn:
        model_name = 'model_2019-08-08_19-15_rejectionBlock_R1_process_2'
        n_test = 1000
        test(model, n_test, model_name)
