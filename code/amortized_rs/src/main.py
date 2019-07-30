
import torch
from model import density_estimator
from time import strftime
import os 
import sys

inputDimR1 = 1        # takes variable 'x' from R1 
outputDimR1 = 1       # takes variable 'r(x)' from R1
inputDimR1 = 2        # takes variable 'x' from R2 
outputDimR1 = 1       # takes variable 'r(x)' from R2
learningRate = 0.01 
epochs = 100

# Create a folder to store experiment results
timestamp = strftime("%Y-%m-%d_%H-%M")
directory = "results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DensityEstimatorTrainer():
    """
    Class to estimate the density. The class takes a pytorch optmizer
    """

    def __init__(self, device, optmizer):
        self.device = device
        self.optmizer = optmizer
        self.steps = 0

    def train(self, epochs, data):
        """
        Method to train the density estimator. 

        :param: epochs :type: int :descrp: Number of epochs to train for.
        :param: data :type: torch.Tensor  :descrp: The data to pass through the network. 

        :return: Trained model
        """

        for epoch in range(epochs):
            epoch_loss = 0
            for 
