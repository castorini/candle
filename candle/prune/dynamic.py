import math
import random

from torch.autograd import Variable
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from candle.quantize.soft import inclusive_clamp
from candle.context import Context
from candle.estimator import *
from candle.nested import *
from candle.proxy import *

def align_mask(x, other):
    other = other.unsqueeze(0)
    for _ in x.size()[2:]:
        other = other.unsqueeze(-1)
    return other.expand_as(x)

class CONCRETEDropout(nn.Module):
    def __init__(self, size, alpha=0, beta=2/3, gamma=-0.1, zeta=1.1, scale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(size).normal_(alpha, std=0.01))
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.size = size
        self.use_scale = scale
        if scale:
            self.scale = nn.Parameter(torch.ones(size))
        self.active = False

    def l0_loss(self):
        return (self.alpha - self.beta * math.log(-self.gamma / self.zeta)).sigmoid().sum()

    def forward(self, x):
        if not self.active:
            return x
        if self.training:
            u = torch.Tensor(self.size).uniform_()
            if x.is_cuda:
                u = u.cuda()
            s = ((u.log() - (1 - u).log() + self.alpha) / self.beta).sigmoid()
            s = s * (self.zeta - self.gamma) + self.gamma
            z = s.clamp(0, 1)
        else:
            z = self.alpha.sigmoid() * (self.zeta - self.gamma) + self.gamma
            z.data.clamp_(0, 1)
            # print((z != 0).float().sum().item(), z.numel())
        z = align_mask(x, z)
        if self.use_scale:
            z = align_mask(x, self.scale) * z
        return x * z

class AlphaDropout(nn.Module):
    def __init__(self, size, init=0.1, bounds=(0.05, 0.99)):
        super().__init__()
        self.size = size
        self.mask = nn.Parameter(torch.Tensor(size).fill_(init))
        self.bounds = bounds
        self.active = False
        self.pruned = False

    def prune(self, keep_pct=None, fixed_size=None):
        prune_pct = 1 - keep_pct
        if fixed_size is None:
            fixed_size = int(prune_pct * self.size)
        else:
            fixed_size = self.size - fixed_size
        self.active = True
        self.pruned = True
        values, indices = torch.sort(self.mask)
        indices = indices[:fixed_size]
        self.mask.data[indices] = 0
        self.mask.data[self.mask.data != 0] = 1
        return indices

    def forward(self, x):
        if not self.active:
            return x
        mask = self.mask
        if not self.pruned:
            mask = mask.clamp(*self.bounds)
            mask = restrict_grad(mask, lambda x: x > 0)
        mask = mask.unsqueeze(0)
        for _ in x.size()[2:]:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(x) * x

class LinearMarkovDropout(nn.Module):
    def __init__(self, end_prob=0, min_length=0, rescale=True, tied=False, 
            tied_generator=None, tied_root=False, rebias=False, size=None):
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

    def forward(self, x, refresh=True):
        if self.tied_root and refresh:
            self.tied_generator.reset()
        min_length = int(self.min_length * x.size(1))
        if self.fixed_size is not None and not self.training:
            if self.rebias and self.fixed_size < self.add_mask.size(0):
                rebias = self.add_mask[self.fixed_size]
                x.add_(rebias)
            x[:, self.fixed_size:, ...] = 0
            return x
        if not self.training and not self.tied_generator.fixed:
            return x
        size = int((x.size(1) - self.min_length) / (1 - self.end_prob))
        if self.tied:
            end_idx = self.tied_generator(min_length, min_length + size)
        else:
            end_idx = random.randint(min_length, min_length + size)
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
        self.t = 1
        self.fixed = False
        self.reset()

    def reset(self):
        if not self.fixed:
            self.t = random.random()

    def fix(self, t):
        self.fixed = True
        self.t = t

    def __call__(self, a, b):
        return a + round(self.t * (b - a))
