from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .nested import *

class Function(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class DifferentiableFunction(Function):
    def diff(self, *args, **kwargs):
        raise NotImplementedError

class ProbabilityDistribution(DifferentiableFunction):
    def draw(self, *args, **kwargs):
        raise NotImplementedError

class BernoulliDistribution(ProbabilityDistribution):
    def __call__(self, b, theta):
        return theta * b + (1 - b) * (1 - theta)

    def diff(self, b, theta):
        return b * 2 - 1

    def draw(self, theta):
        return theta.clamp(0, 1).bernoulli()

class GradientEstimator(Function):
    def estimate_gradient(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.estimate_gradient(*args)

# class SimpleBernoulliReparameterization(ProbabilityDistribution):
#     def __call__(self, )

"""
Implements RELAX estimator from "Backpropagation Through the Void..." paper
https://arxiv.org/abs/1711.00123
Grathwohl et al. (2018)
"""
class RELAXEstimator(GradientEstimator):
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

class REINFORCEEstimator(GradientEstimator):
    def __init__(self, f, p):
        self.f = f
        self.p = p

    def estimate_gradient(self, theta):
        b = self.p.draw(theta)
        return self.f(b) * self.p.diff(b) / self.p(b, theta)

"""
REINFORCE with Importance Sampling Estimator
"""
class RISEEstimator(GradientEstimator):
    def __init__(self, f, p, p_i, alpha=0.95):
        self.f = f
        self.p = p
        self.p_i = p_i
        self.alpha = alpha
        self._running_mean = None
        self._running_diff = None

    def _update_running_mean(self, mean):
        if self._running_mean is None:
            self._running_mean = mean
            return
        self._running_mean = self.alpha * self._running_mean + (1 - self.alpha) * mean

    def _update_running_diff(self, diff):
        if self._running_diff is None:
            self._running_diff = diff
            return
        self._running_diff = self.alpha * self._running_diff + (1 - self.alpha) * diff

    def _compute_diff(self, b, theta, pi):
        weight = self.p.diff(b, theta) * self.f(b)
        return weight * (-1 / (self.p_i(b, pi)**2 + 1E-6)) * self.p_i.diff(b, pi)

    def _compute_var_grad(self, b, theta, pi, gis):
        diff = self._compute_diff(b, theta, pi)
        self._update_running_diff(diff)
        self._update_running_mean(gis)
        pt1 = gis - self._running_mean
        pt2 = diff - self._running_diff
        return 2 * pt1 * pt2

    def estimate_gradient(self, theta, pi):
        b = self.p_i.draw(pi)
        gis = self.f(b) * self.p.diff(b, theta) / (self.p_i(b, pi) + 1E-6)
        var_grad = self._compute_var_grad(b, theta, pi, gis)
        return gis, var_grad
