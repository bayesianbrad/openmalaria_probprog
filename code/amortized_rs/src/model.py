
import torch
from torch import nn
from torch.nn import functional as F

class density_estimator(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(density_estimator, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


