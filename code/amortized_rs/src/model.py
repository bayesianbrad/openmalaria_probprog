
import torch
from torch import nn
from torch.nn import functional as F

class density_estimator(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(density_estimator, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(inputSize, inputSize//2),
            nn.ReLU(),

        )
        self.linear_mu = nn.Linear(inputSize, outputSize)
        self.linear_std = nn.Linear(inputSize, outputSize)

    def forward(self, x):

        return self.linear_mu(x), nn.Softplus(self.linear_std(x))+1e-6


