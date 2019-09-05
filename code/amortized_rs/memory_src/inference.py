import torch.multiprocessing as mp
from torch.utils.cpp_extension import load
import torch as th
import torch.nn.functional as F
from torch import optim
import torch.distributions as dist
from time import strftime
import os
import time
import shutil
import logging
import argparse
import seaborn as sns
import warnings
import math
import importlib
import sys
import datetime
import ast
import seaborn as sns



class Inference():
    """
    Class for training and testing the regressor.
    """

    def __init__(self, parameters):
        '''

        :param address: :type str Which rejection sampling blocks to amortize
        :param batchSize: :type int Number of samples to push at each iteration.
        :param model: :type class Neural network architecture 
        :param proposal: :type A class representing a proposal function
        :param optimizerparams :type dict Optimizer params: 'name
        :param loss: :type torch.loss Pytorch loss function
        :param nIterations: :type int Number of iterations for training
        :param device: :type str Device to be used for data generation
        :param loadData: :type bool determines whether to upload pre-loaded data (true), or generate data on the fly (False)
        :param loadDatafName :type str load file name
        :param logPath :type str log save path
        :param loadCheckpoint :type File to load to resume from checkpoint.
        :param trainon :type bool Train netowrk to amortize address
        :param teston :type bool Test learnt network.
        :param modelname :type str PATH to existing model
        :param testiterations :type int Number of test epochs
        :param savedata :type bool Save data
        '''
        self.amortized_rs = load(name="amortized_rs",
                            sources=["amortized_rs.cpp"])
        self.address = parameters.address
        self.batchSize = parameters.batchsize
        self.model = parameters.model
        self.optimizerParams = parameters.optimparams
        self.optimizerParams = ast.literal_eval(self.optimizerParams)
        # generate optimizer
        self.testOn = parameters.teston
        if self.testOn:
            try:
                self.testIterations = parameters.testiterations
            except:
                ValueError

        self.trainOn = parameters.trainon
        # self.optimizer = optimizer
        if parameters.proposal:
            self.proposalModule =  importlib.import_module('proposal')
            self.proposalClass = getattr(self.proposalModule, 'Proposal')()
            self.proposalMethod = parameters.proposal
        # self.loss = parameters.loss
        self.nIterations = parameters.trainiterations
        self.device = parameters.device
        self.loadData = parameters.loaddata
        self.logPath = parameters.logpath
        self.saveData = parameters.savedata
        if parameters.loadcheckpoint:
           logging.info("Restoring parameters from {}".format('../checkpoint/'+parameters.loadCheckpoint))
           self.loadCheckpoint(parameters.loadCheckpoint)
        if parameters.ncores == math.inf:
            self.processes = mp.cpu_count()
        else:
            self.processes = parameters.ncores
        if self.address == 'R2':
            self.inputsize = 2*self.batchSize
            self.outputsize = self.batchSize
        if self.address == 'R1' or 'R3':
            self.inputsize = self.batchSize
            self.outputsize = self.batchSize
        if parameters.model:
            self.modelModule = importlib.import_module('model')
            self.model = getattr(self.modelModule, parameters.model)(self.inputsize, self.outputsize, self.batchSize)
        elif parameters.modelname:
            self.modelName = '../model/' + parameters.modelName
            self.model = self.model.load_state_dict(th.load(self.modelName))
        else:
            warnings.warn('*****Model must be specified*****')

        # intialize optimizer
        self.optimizer_Fn(self.optimizerParams)
        dateStr = str(datetime.datetime.now())
        self.dataPath = '../data/{}/'.format(dateStr)
        if self.saveData:
            if not os.path.exists(self.dataPath):
                os.makedirs(self.dataPath)



    def optimizer_Fn(self,optimizerParams):
        '''
        Sets-up the optimizer

        '''
        if optimizerParams['name'] == 'Adadelta':
            self.optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=optimizerParams['lr'] if optimizerParams['lr'] else 0.01,
                                               rho=optimizerParams['rho'] if optimizerParams['rho'] else 0.9,
                                               eps=optimizerParams['eps'] if optimizerParams['eps'] else 1e-6,
                                               weight_decay=optimizerParams['weight_decay'] if optimizerParams['weight_decay'] else 0)
        if optimizerParams['name'] == 'Adagrad':
            self.optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=optimizerParams.get('lr', 0.01),
                                               lr_decay=optimizerParams.get('lr_decay', 0),
                                               weight_decay=optimizerParams.get('weight_decay', 0),
                                               initial_accumulator_value=optimizerParams.get('initial_accumulator_value', 0))
        if optimizerParams['name'] == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=optimizerParams.get('lr',0.001),
                                               betas=optimizerParams.get('betas',(0.9,0.999)),
                                               eps= optimizerParams.get('eps', 1e-8),
                                               weight_decay=optimizerParams.get('weight_decay',0),
                                               amsgrad=optimizerParams.get('amsgrad',False))
        if optimizerParams['name'] == 'AdamW':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=optimizerParams.get('lr',0.001),
                                               betas=optimizerParams.get('betas',(0.9,0.999)),
                                               eps= optimizerParams.get('eps', 1e-8),
                                               weight_decay=optimizerParams.get('weight_decay',0),
                                               amsgrad=optimizerParams.get('amsgrad',False))
        #TODO change everything to <dict_name>.get
        if optimizerParams['name'] == 'SparseAdam':
            self.optimizer = optim.SparseAdam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=optimizerParams['lr'] if optimizerParams['lr'] else 0.001,
                                           betas=optimizerParams['betas'] if optimizerParams['betas'] else (
                                           0.9, 0.999),
                                           eps=optimizerParams['eps'] if optimizerParams['eps'] else 1e-8)
        if optimizerParams['name'] == 'Adamax':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=optimizerParams['lr'] if optimizerParams['lr'] else 0.002,
                                           betas=optimizerParams['betas'] if optimizerParams['betas'] else (
                                           0.9, 0.999),
                                           eps=optimizerParams['eps'] if optimizerParams['eps'] else 1e-08,
                                           weight_decay=optimizerParams['weight_decay'] if optimizerParams[
                                               'weight_decay'] else 0)
        if optimizerParams['name'] == 'ASGD':
            self.optimizer = optim.ASGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=optimizerParams['lr'] if optimizerParams['lr'] else 0.01,
                                           lambd=optimizerParams['lambd'] if optimizerParams['lambd'] else 1e-4,
                                           alpha=optimizerParams['alpha'] if optimizerParams['alpha'] else 0.75,
                                           t0=optimizerParams['t0'] if optimizerParams['t0'] else 1e6,
                                           weight_decay=optimizerParams['weight_decay'] if optimizerParams[
                                               'weight_decay'] else 0)
        # ignoring LBFGS as it is not well supported in pyTorch, they are working on it though. TODO: Revise at a later date

        if optimizerParams['name'] == 'RMSprop':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=optimizerParams['lr'] if optimizerParams['lr'] else 0.01,
                                           momentum=optimizerParams['momentum'] if optimizerParams['momentum'] else 0,
                                           alpha=optimizerParams['alpha'] if optimizerParams['alpha'] else 0.99,
                                           eps=optimizerParams['eps'] if optimizerParams['eps'] else 1e-08,
                                           centered=optimizerParams['centered'] if optimizerParams['centered'] else False,
                                           weight_decay=optimizerParams['weight_decay'] if optimizerParams[
                                               'weight_decay'] else 0)

        if optimizerParams['name'] == 'Rprop':
            self.optimizer = optim.Rprop(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=optimizerParams['lr'] if optimizerParams['lr'] else 0.01,
                                              etas=optimizerParams['etas'] if optimizerParams[
                                                  'etas'] else (0.5,1.2),
                                              step_sizes=optimizerParams['step_sizes'] if optimizerParams[
                                                  'step_sizes'] else (1e-6,50))

        if optimizerParams['name'] == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=optimizerParams.get('lr',0.01),
                                              momentum=optimizerParams.get('momentum',0),
                                              dampening=optimizerParams.get('dampening', 0.0),
                                              weight_decay=optimizerParams.get('weight_decay', 0))

    # no longer need write the batch loop in c - we can do that later
    def get_batch(self, count, iteration=None):
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
        if self.loadData is None:
            print('{0} Generating data {0}'.format(5*'='))
            data = th.stack([self.amortized_rs.f() for _ in range(self.batchSize)])
            print('{0} Generating completed {0}'.format(5 * '='))
            if self.saveData:
                fname = self.dataPath + str(iteration)+'_samples_.th'
                th.save(data,fname)

        if self.address == 'R1' and not self.loadData:
            inR1 = data[0:self.batchSize, 0].view([self.batchSize, 1]).view([1, self.batchSize])
            outR1 = data[0:self.batchSize, 1].view([self.batchSize, 1]).view([1, self.batchSize])
            count = count + 1  # not actually need if self.loadData == True
            return inR1.to(self.device), outR1.to(self.device), count
        elif self.address == 'R1' and self.loadData:
            inR1 = data[count:self.batchSize + count * self.batchSize, 0]
            outR1 = data[count:self.batchSize + count * self.batchSize, 1]
            count = count + 1
            return inR1.to(self.device), outR1.to(self.device), count
        elif self.address == 'R2' and not self.loadData:
            inR2 = th.cat([data[0:self.batchSize, 1], data[0:self.batchSize, 2]], dim=0).view(1, 256)
            outR2 = data[0:self.batchSize, 3].view([1, self.batchSize])
            count = count + 1
            return inR2.to(self.device), outR2.to(self.device), count
        elif self.address == 'R2' and self.loadData:
            inR2 = th.stack(
                [data[count:self.batchSize + count * self.batchSize, 1], data[count:self.batchSize + count * self.batchSize, 2]], dim=1)
            outR2 = data[count:self.batchSize + count * self.batchSize, 1]
            count = count + 1
            return inR2.to(self.device), outR2.to(self.device), count


    def train(self, process=1, saveModel=True, saveName=None, checkpoint=True):
        '''
        Method to train model

        :param process: :type int Process number, when used in parallel.
        :param saveModel :type bool To save model set True.
        :param saveName :type str Default saveName for model is a time stamp + process name.
        :param checkpoint :type bool If you want to store avg loss data, for each iteration, model weights etc
        :return:
        '''

        #TODO: Add device option to transfer to the preset device.
        print('{0} Training initiated {0}'.format(5*'='))
        self.model.train()
        self.optimizer.zero_grad()
        count = 0
        _outLoss = 0

        # The network needs to learn z1 -> z2 and z2,z3 -> z4
        for i in range(self.nIterations):
            print('Debug statement : self.train()  Interation number {}'.format(i))
            inData, outData, count = self.get_batch(count, iteration=i)
            print('Debug statement : self.train()  data generated for iteration '.format(i))
            proposal = getattr(self.proposalClass,self.proposalMethod)(inData=inData, batchSize=self.batchSize, model=self.model)
            print('Debug statement : self.train()  proposal generated for iteration '.format(i))
            self.optimizer.zero_grad()
            _loss = -proposal.log_prob(outData)

            # calculate the SVI update sum of log(prob..) / 'batchsize'
            totalLoss = _loss.sum() / len(self.batchSize)
            totalLoss.backward()
            self.optimizer.step()
            # if using a a learning rate scheduler place below optimizer.step() (if pytorch version >= 1.1.0
            if i == 0:
                bestLoss = totalLoss.item()
                bestFlag = True
            if i > 0:
                if bestLoss < totalLoss.item():
                   bestFlag = False
                else:
                    bestFlag = True
                    bestLoss = totalLoss.item()
            _outLoss += totalLoss.item()

            if i % 100 == 0:
                avgLoss = _outLoss / (i + 1)
                print("iteration {}: Average loss: {} Iteration loss: {}".format(i,avgLoss, totalLoss.item()))
                if checkpoint:
                    checkpointData = {'model': self.modelModule,
                                      'state_dict': self.model.state_dict(),
                                      'iteration': i,
                                      'optimizer': self.optimizer.state_dict(),
                                      'addresss': self.address,
                                      'avg_loss': avgLoss,
                                      'iter_loss':totalLoss.item()
                                      }
                    self.save_checkpoint(checkpointData, bestFlag,saveName, process, self.address)

        if not os.path.exists('../model/'):
            os.makedirs('../model/')
        if saveModel:
            if saveName:
                fname = '../model/{}_process_{}_address_{}'.format(saveName, process, self.address)
            else:
                fname = '../model/{}_process_{}_address_{}'.format(strftime("%Y-%m-%d_%H-%M"), process, self.address)
            th.save(self.model.state_dict(), fname)
            print(' Model is saved at : {}'.format(fname))
            self.modelName = fname

    def save_checkpoint(self, state, bestFlag, saveName, process, address):
        """
        Saves model and training parameters at checkpoint + '<name>.pth.tar'. If the loss is the lowest,
        it saves the model as 'bestModel_<name>.pth.tar' and will overwrite any best models before it.

        :param:state: :type dict Contains model's state, may contain other keys such as epoch, optimizer
        :param: bestFlag: :type bool Was this models loss lower than the last.
        :param: saveName: :type str or None.Name under which to save the model
        :param: process: :type int Which thread
        :param: address: :type str Which address we are learning a surragote for.
        """
        if not os.path.exists('../checkpoints/'):
            os.mkdir('../checkpoints/')
        if saveName:
           name ='{}_{}_{}'.format(saveName,process,address)
        else:
           name = strftime("%Y-%m-%d_%H-%M") + '_{}_{}'.format(process,address)
        if bestFlag and saveName:
           name ='bestModel_{}_{}_{}'.format(saveName,process,address)
        elif bestFlag:
           # assuming jobs will take less than 1 day to run
           name ='bestModel_{}_{}_{}'.format(strftime("%Y-%m-%d"), process, address)

        fName = '../checkpoints/' + os.path.join(name, '.pth.tar')
        th.save(state, fName)
        print('{0} Saved checkpoint under file name : {1} {0}'.format(5*'*', fName))
    def load_checkpoint(self, checkpoint):
        """Loads model parameters (state_dict) from file_path.

        :param: checkpoint: :type str Filename which needs to be loaded
        :param: model: :type torch.nn.Module  Model for which the parameters are loaded
        :param: optimizer: :type torch.optim  Optional: resume optimizer from checkpoint
        """
        # assumes only file name is passed.
        checkpoint = '../checkpoints/'+checkpoint+'pth.tar'
        if not os.path.exists(checkpoint):
            raise ("File doesn't exist {}".format(checkpoint))
        checkpointData = th.load(checkpoint)
        self.model = self.model.load_state_dict(checkpointData['state_dict'])
        self.optimizer = self.optimizer.load_state_dict(checkpointData['optimizer'])
        self.address = checkpointData['address']

        return checkpointData

    def test(self, testSamples):
        '''
        Test learnt proposal

        :param testSamples: :type int Number of samples to generate from learnt proposal
        :param modelName: :type str model name which to load.
        :return:
        '''
        self.modelTest = self.model.load_state_dict(th.load(self.modelName))
        self.modelTest.eval()
        totalKL = 0
        with th.no_grad():
            for i in range(testSamples):
                inData, outData, count = self.get_batch(self.address, testSamples,cout=0)
                # may have to change this api for generating samples from the proposal later on.
                # An efficient, analytically exact sample method must be implemented

                learntProposal  = self.proposal(*self.modelTest(inData)).sample()
                # If we have learnt a good proposal then the KL will tend to zero.
                # I think this is right TODO: Check KL code later
                kl = F.kl_div(outData, learntProposal, reduction='batchmean')
                totalKL += kl
                if i % 100 == 0:
                    print('{} Iteration , Test avg KL: {}\n'.format(i + 1, totalKL / (i + 1)))
                    self.plot(pred=learntProposal,generatedData=outData,iteration=i,rank=self.processes)



    def plot(self, pred=None, generatedData=None, iteration=None, rank=1):
        '''
        Plot the predicted density agaisnt the simulated data
        :param pred:
        :param generatedData:
        :param iteration:
        :param rank:
        :return:
        '''

        ax= sns.distplot(pred[0, :].detach().numpy(), kde=True, color='r',label='pred')
        ax = sns.distplot(generatedData[0, :].detach().numpy(), kde=True, color='b', label='ground truth')
        ax.legend()
        ax.set_title('Iteration_{}_Rejection_block_{}'.format(iteration,self.address))
        fig = ax.get_figure()
        if os.path.exists('../plots/'):
            os.makedirs('../plots/')
        fname= '../plots/{}_compare_pred_vs_gt_iteration_rejection_block_{}.png'.format(strftime("%M:%S"),self.address)
        fig.savefig(fname=fname)

        plot_pred = sns.distplot(pred[0, :].detach().numpy(), kde=True, color='r')
        plot_true = sns.distplot(generatedData[0, :].detach().numpy(), kde=True, color='b')
        fnamePred = '../plot/' + 'predicted_iteration_{}_process_{}_rejectionBlock_{}'.format(iteration, rank, self.address)
        fnameTrue = '../plot/' + 'true_iteration_{}_process_{}_rejectionBlock_{}'.format(iteration, rank, self.address)
        plot_pred.savefig()
    def run(self,*args, **kwargs):
        '''
        Runs the methods
        :param args:
        :param kwargs:
        :return:
        '''

        if self.trainOn:
            keywords = {'saveModel': kwargs['saveModel'],
                        'saveName': kwargs['saveName'],
                        'checkpoint': kwargs['checkpoint']}
            if self.processes == 1:
                self.train(self.processes, keywords)
            else:
                self.model.share_memory()
                processes = []
                for process in range(self.processes):
                    p = mp.Process(target=self.train, args=(process), kwargs=keywords)
                    p.start()
                    processes.append(p)
                for p in self.processes:
                    p.join()


        if self.testOn:
            self.test(self.testIterations)

def main():
    try:
        parser = argparse.ArgumentParser(description='Amortized sub-programs ',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--device', '--d', help='Set the compute device (cpu, cuda:0, cuda:1, etc.)', default='cpu',
                            type=str)
        parser.add_argument('--seed', help='Random number seed', default=None, type=int)
        parser.add_argument('--address', '--a', help='Addreess to be amortized', default=None, type=str)
        parser.add_argument('--batchsize', '--bs', help='Number of samples to pass into trainer', default=128, type=int)
        parser.add_argument('--optimparams', '--op', type=str)
        parser.add_argument('--model', '--m', help='The name of the model to import from a local directory',
                            default='density_estimator', choices=['density_estimator'], type=str)
        # parser.add_argument('--inputsize', '--is', help='size of inputs into model', default=128, type=int)
        # parser.add_argument('--outputsize', '--os', help='size of outputs of model', default=128, type=int)
        parser.add_argument('--trainon', help='If you want to perform amortized inference training, default True"', default=True,
                            type=bool)
        parser.add_argument('--trainiterations', '--ntr', help='The number of train iterations. Default 1000',
                            default=1000, type=int)
        parser.add_argument('--teston', help='If you want to test your model. Default False', default=False, type=bool)
        parser.add_argument('--testiterations', '--nti', help='The number of test iterations. Default 1000', default=1000, type=int)
        parser.add_argument('--loadcheckpoint', '--lp', help='PATH to load checkpoint "../checkpoints/"', default=None,
                            type=str)

        parser.add_argument('--modelname', '--mn', help='PATH to existing model "../mdoel/<modelname>"', default=None,
                            type=str)
        parser.add_argument('--ncores', help='N cores to utilize. The default is all cores', default=math.inf,type=int)
        parser.add_argument('--proposal', '--pr', help='the name of the objective to use deault NormalApproximator', type=str)
        # parser.add_argument('--loss', '--l', help='The name of the loss function to use', default=None, type=str)
        parser.add_argument('--loaddata', '--ld', help='PATH to load data from', default=None, type=str)
        parser.add_argument('--logpath', help='PATH to save log', default=None, type=str)
        parser.add_argument('--savename', '--sn', help='File name to save model to "../model/<savename>"', default=None,
                            type=str)
        parser.add_argument('--savemodel', '--sm', help='If you want the model saved, set to True, default True', default=True,
                            type=bool)
        parser.add_argument('--checkpoint', '--chk', help='Ifmain you want to save checkpoints. Default is True.  ', default= True, type=bool)
        parser.add_argument('--savedata', '--sd', help='True if you want to save data, else False. Default True',default=True, type=bool )
        opt = parser.parse_args()

        inference =Inference(opt)
        inference.run(saveModel=opt.savemodel, saveName=opt.savename, checkpoint=opt.checkpoint)


    except KeyboardInterrupt:
        print('Stopped.')



if __name__ == '__main__':
    time_start = time.time()
    main()
    print('\nTotal duration: {}'.format((time.time() - time_start)))
    sys.exit(0)
