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
from .soft import *

def bimodal_xavier_normal(x, a_loc, b_loc, a_gain=1, b_gain=1):
    unsqueezed = False
    old_x = x
    if x.dim() == 1:
        unsqueezed = True
        x = x.unsqueeze(1)
    u = x.uniform_()
    mask1 = (u < 0.5).float()
    mask2 = (u >= 0.5).float()
    x.copy_(mask1 * (nn.init.xavier_normal(x, a_gain) + a_loc) + \
        mask2 * (nn.init.xavier_normal(x, b_gain) + b_loc))
    if unsqueezed:
        old_x.copy_(x.squeeze(1))
    return old_x

class StochasticQuantizeHook(ProxyDecorator):
    def __init__(self, layer, child, t=0.8, limit=None):
        super().__init__(layer, child)
        self.t = t if isinstance(t, nn.Parameter) else nn.Parameter(torch.Tensor([t]))
        self.clamp = None if limit is None else (-limit, limit)
        self.layer.init_weights(lambda x: bimodal_xavier_normal(x, -2, 2))

    def sizes(self):
        return self.child.sizes

    def parameters(self):
        return [self.t]

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        if self.clamp:
            input.data.clamp_(*self.clamp)
        self.t.data.clamp_(min=0)
        scale = restrict_grad(self.t, lambda x: x < 0)
        input = input.apply_fn(lambda x: 2 * sample_bernoulli_concrete(x.exp(), scale, 
            training=self.layer.training) - 1)
        return input

class StochasticTanhFunction(ag.Function):
    @staticmethod
    def forward(ctx, x):
        return 2 * x.bernoulli() - 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class StochasticActivation(nn.Module):
    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def forward(self, x):
        if self.limit is not None:
            x = restrict_grad(x, lambda x: (x < -self.limit) | (x > self.limit))
        x = stochastic_tanh(x.sigmoid())
        return x

class StochasticQuantizeContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def list_scale_params(self, inverse=False):
        check = lambda x: isinstance(x, StochasticQuantizeHook) or isinstance(x.layer, StochasticActivation)
        if inverse:
            return super().list_params(lambda proxy: not check(proxy))
        return super().list_params(check)

    def list_model_params(self):
        return self.list_scale_params(True)

    def activation(self, limit=None):
        return self.bypass(StochasticActivation(limit=limit))

    def compose(self, layer, limit=None, scale=0.8, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(StochasticQuantizeHook, t=scale, limit=limit)
        return layer

stochastic_tanh = StochasticTanhFunction.apply
