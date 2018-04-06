import gc
import copy

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

class TernaryClampFunction(ag.Function):
    @staticmethod
    def forward(ctx, x):
        delta = 0.7 * x.abs().mean()
        x = x / delta
        return x.clamp(-1, 1).int().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantizeKFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, k):
        factor = (2**k - 1)
        return 1 / factor * (factor * x).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def tanh_quantize(x, k):
    return 2 * quantize_k(x.tanh() / (2 * x.abs().max()) + 0.5, k) - 1

ternary_clamp = TernaryClampFunction.apply
quantize_k = QuantizeKFunction.apply

class TernaryWeightHook(ProxyDecorator):
    def __init__(self, layer, child):
        super().__init__(layer, child)

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        return input.apply_fn(ternary_clamp)

class DoReFaActivation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return tanh_quantize(x, self.k)

class TernaryQuantizeContext(Context):
    def __init__(self, scale_alpha=0, soft=True, **kwargs):
        super().__init__(**kwargs)
        self.soft = soft
        self.scale_alpha = scale_alpha

    def activation(self, k=2):
        return self.bypass(DoReFaActivation(k))

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(TernaryWeightHook)
        return layer
