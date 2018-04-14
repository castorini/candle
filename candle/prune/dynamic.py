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

def align_mask(x, other):
    other = other.unsqueeze(0)
    for _ in x.size()[2:]:
        other = other.unsqueeze(-1)
    return other.expand_as(x)

class LinearMarkovDropout(nn.Module):
    def __init__(self, end_prob=0, min_length=0, variational=False, rescale=False):
        super().__init__()
        self.end_prob = end_prob
        self.variational = variational
        self.fixed_size = None
        self.min_length = min_length
        self.rescale = rescale
        self._x_cache = None

    def fix(self, fixed_size):
        self.fixed_size = fixed_size
        return self

    def forward(self, x):
        min_length = int(self.min_length * x.size(1))
        if self.fixed_size is not None and not self.training:
            x[:, self.fixed_size:, ...] = 0
            return x
        if not self.training:
            return x
        size = int((x.size(1) - self.min_length) / (1 - self.end_prob))
        if self.variational:
            for idx in range(x.size(0)):
                end_idx = random.randint(min_length, min_length + size - 1)
                x.data[idx, end_idx:, ...] = 0
                if self.rescale:
                    x[idx, ...] *= x.size(1) / end_idx
        else:
            end_idx = random.randint(min_length, min_length + size - 1)
            x = x.clone()
            x[:, end_idx:, ...] = 0
            if self.rescale:
                if self._x_cache is None:
                    self._x_cache = torch.arange(1, self.end_prob, -(1 - self.end_prob) / (x.size(1) - min_length))
                    if x.is_cuda:
                        self._x_cache = self._x_cache.cuda()
                if end_idx - min_length == 0:
                    return x
                rescale = self._x_cache[:end_idx - min_length]
                rescale = rescale.unsqueeze(0)
                for _ in x.data.size()[2:]:
                    rescale = rescale.unsqueeze(-1)
                x[:, min_length:end_idx, ...] /= rescale.expand_as(x[:, min_length:end_idx, ...])
        return x
