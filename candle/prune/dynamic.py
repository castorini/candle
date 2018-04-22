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
    def __init__(self, end_prob=0, min_length=0, rescale=True, tied=False, 
            tied_generator=None, tied_root=False, rebias=True, size=None):
        super().__init__()
        self.end_prob = end_prob
        self.fixed_size = None
        self.min_length = min_length
        self.rescale = rescale
        self._x_cache = None
        self.tied = tied
        self.tied_generator = tied_generator
        self.tied_root = tied_root
        self.rebias = rebias
        if rebias:
            self.add_mask = nn.Parameter(torch.zeros(size))

    def fix(self, fixed_size):
        self.fixed_size = fixed_size
        return self

    def forward(self, x):
        if self.tied_root:
            self.tied_generator.reset()
        min_length = int(self.min_length * x.size(1))
        if self.fixed_size is not None and not self.training:
            if self.rebias and self.fixed_size < self.add_mask.size(0):
                rebias = self.add_mask[self.fixed_size]
                x.add_(rebias)
            x[:, self.fixed_size:, ...] = 0
            return x
        if not self.training:
            return x
        size = int((x.size(1) - self.min_length) / (1 - self.end_prob))
        if self.tied:
            end_idx = self.tied_generator(min_length, min_length + size - 1)
        else:
            end_idx = random.randint(min_length, min_length + size - 1)
        z = x.clone().detach()
        z.fill_(1)
        if self.rebias and end_idx < self.add_mask.size(0):
            rebias = self.add_mask[end_idx]
            x = x + rebias
        z[:, end_idx:, ...] = 0
        x = x * z
        if self.rescale:
            if self._x_cache is None:
                ones = torch.ones(min_length)
                arange = torch.arange(1, self.end_prob, -(1 - self.end_prob) / (x.size(1) - min_length))
                self._x_cache = torch.cat([ones, arange])
                if x.is_cuda:
                    self._x_cache = self._x_cache.cuda()
            if end_idx - min_length == 0:
                return x
            rescale = self._x_cache
            rescale = rescale.unsqueeze(0)
            for _ in x.data.size()[2:]:
                rescale = rescale.unsqueeze(-1)
            x = x / rescale.expand_as(x)
        return x

class UniformTiedGenerator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = random.random()

    def __call__(self, a, b):
        return a + round(self.t * (b - a))
