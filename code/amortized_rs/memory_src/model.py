import torch
from torch import nn
from torch.nn import functional as F

class density_estimator(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(density_estimator, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(inputSize, inputSize // 2, bias=True),
            nn.ReLU(),
            nn.Linear(inputSize // 2, inputSize // (inputSize // 2), bias=True)
            # ensures that the output would be 2, assuming inputsize = 2^n
        )
        self.linear_mu = nn.Linear(inputSize // (inputSize // 2), outputSize)
        self.linear_std = nn.Linear(inputSize // (inputSize // 2), outputSize)

    def forward(self, x):
         # x is of shape [batch_size, input_dim]
        hidden = self.hidden(x) # return [batch_size//2, 2]
        mu = self.linear_mu(hidden)
        sigma = F.softplus(self.linear_std(hidden)) + 1e-6
        return mu, sigma

