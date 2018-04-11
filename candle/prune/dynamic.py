import random

from torch.autograd import Variable
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from candle.context import Context
from candle.estimator import *
from candle.nested import *
from candle.proxy import *

class DynamicDropout(nn.Module):
    def __init__(self, end_prob=0.1, variational=False):
        super().__init__()
        self.end_prob = end_prob
        self.variational = variational
        self.fixed_size = None

    def fix(self, fixed_size):
        self.fixed_size = fixed_size
        return self

    def forward(self, x):
        if self.fixed_size is not None and not self.training:
            x[:, self.fixed_size:, ...] = 0
            return x
        if not self.training:
            return x
        size = int(x.size(1) / (1 - self.end_prob))
        if self.variational:
            for idx in range(x.size(0)):
                end_idx = min(random.randint(0, size - 1), x.size(1))
                x[idx, end_idx:, ...] = 0
        else:
            end_idx = min(random.randint(0, size - 1), x.size(1))
            x[:, end_idx:, ...] = 0
        return x
