import gc

from torch.autograd import Variable
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .nested import *

class Function(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class RebarFunction(Function):
    def __init__(self, function, temp):
        self.function = function
        self.temp = temp
        self.concrete_fn = ConcreteRelaxation(temp)

    def __call__(self, theta, noise):
        return self.function(self.concrete_fn(theta))

class ProbabilityDistribution(Function):
    def draw(self, *args, **kwargs):
        raise NotImplementedError

class BernoulliDistribution(ProbabilityDistribution):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, b):
        return (self.theta + 1E-8)**b * (1 - self.theta + 1E-8)**(1 - b)

    def draw(self):
        return self.theta.clamp(0, 1).bernoulli()

class SoftBernoulliDistribution(ProbabilityDistribution):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, b):
        return (self.theta.sigmoid() + 1E-8)**b * (1 - self.theta.sigmoid() + 1E-8)**(1 - b)

    def draw(self):
        return self.theta.sigmoid().bernoulli()

class GradientEstimator(Function):
    def estimate_gradient(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.estimate_gradient(*args)

class ConcreteRelaxation(Function):
    def __init__(self, temp):
        self.temp = temp

    def __call__(self, z):
        return (z / self.temp).sigmoid()

class BernoulliRelaxation(ProbabilityDistribution):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, b):
        raise NotImplementedError

    def draw(self):
        theta = self.theta.sigmoid()
        u = theta.clone().uniform_()
        l1 = theta.log() - (1 - theta).log()
        l2 = u.log() - (1 - u).log()
        return l1 + l2, u

class ConditionedBernoulliRelaxation(BernoulliRelaxation):
    def __call__(self, b):
        raise NotImplementedError

    def draw(self, b):
        theta = self.theta.sigmoid()
        v = theta.clone().uniform_()
        t1 = 1 - theta
        v = (v * t1) * (1 - b) + (v * theta + t1) * b
        l1 = theta.log() - t1.log()
        l2 = v.log() - (1 - v).log()
        return l1 + l2, v

class Heaviside(Function):
    def __call__(self, x):
        return x.clamp(0, 1).ceil()

class Round(Function):
    def __call__(self, x):
        return x.round()

class REINFORCEEstimator(GradientEstimator):
    def __init__(self, f, p):
        self.f = f
        self.p = p

    def estimate_gradient(self, theta, b=None):
        b = self.p.draw() if b is None else b
        p = self.p(b)
        f_b = self.f(b)
        dlogp_dtheta = theta.apply_fn(lambda x, out: ag.grad([out.sum()], [x])[0], p.log())
        return dlogp_dtheta * f_b

"""
REINFORCE with Importance Sampling Estimator
"""
class RISEEstimator(GradientEstimator):
    def __init__(self, f, p, p_i, transform_fn=None):
        self.f = f
        self.p = p
        self.p_i = p_i
        self.transform_fn = transform_fn

    def estimate_gradient(self, theta, pi, b=None):
        b = self.p_i.draw() if b is None else b
        h = b.apply_fn(self.transform_fn) if self.transform_fn else b
        p = self.p(h)
        p_i = self.p_i(b)
        f_b = self.f(h)
        dp_dtheta = theta.apply_fn(lambda x, out: ag.grad([out.sum()], [x])[0], p)

        g_is = dp_dtheta * f_b / (p_i + 1E-8)
        var_grad = pi.apply_fn(lambda x, out: ag.grad([out.sum()], [x], retain_graph=True)[0], g_is**2)
        return g_is, var_grad

"""
Implements RELAX estimator from "Backpropagation Through the Void..." paper
https://arxiv.org/abs/1711.00123
Grathwohl et al. (2018)
"""
class RELAXEstimator(GradientEstimator):
    def __init__(self, f, c, p, z, z_tilde, H, transform_fn=None):
        self.f = f
        self.c = c
        self.p = p
        self.z = z
        self.z_tilde = z_tilde
        self.transform_fn = transform_fn
        self.H = H

    def estimate_gradient(self, theta, phi):
        z, u = self.z.draw()
        b = self.H(z)
        zt, v = self.z_tilde.draw(b)
        p = self.p(b)
        c_phi_zt = self.c(theta, v)
        dlogp_dtheta = theta.apply_fn(lambda x, out: ag.grad([out.sum()], [x], retain_graph=True)[0], p.log())

        c_phi_z = self.c(theta, u)
        dc_phi_z = theta.apply_fn(lambda x: ag.grad([c_phi_z], [x], retain_graph=True)[0])
        dc_phi_zt = theta.apply_fn(lambda x: ag.grad([c_phi_zt], [x], retain_graph=True)[0])

        g_relax = dlogp_dtheta * (self.f(b) - c_phi_zt) + dc_phi_z - dc_phi_zt
        var_estimate = (g_relax**2)
        phi_grad = phi.apply_fn(lambda x, y: ag.grad([y.sum()], [x], retain_graph=True)[0], var_estimate)
        gc.collect()
        return g_relax, phi_grad
