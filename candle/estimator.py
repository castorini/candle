from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .nested_list import *

class DiffFunction(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def diff(self, *args, **kwargs):
        raise NotImplementedError

class ProbabilityDistribution(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def draw(self, *args, **kwargs):
        raise NotImplementedError

"""
Implements RELAX estimator from "Backpropagation Through the Void..." paper
https://arxiv.org/abs/1711.00123
"""
class RELAXEstimator(object):
    def __init__(self, f, c, p, z, z_tilde, H):
        self.f = f
        self.c = c
        self.p = p
        self.z = z
        self.z_tilde = z_tilde
        self.H = H

    def estimate_gradient(self, theta):
        z = self.z.draw(theta)
        b = self.H(z)
        z_tilde = self.z_tilde.draw(theta, b)
        dlogp = self.p.diff(b, theta) / self.p(b, theta)
        return (self.f(b) - self.c(z_tilde)) * dlogp + self.c.diff(z) - self.c.diff(z_tilde)

