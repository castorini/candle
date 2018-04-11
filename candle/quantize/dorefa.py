import gc
import copy

from scipy.special import erfcinv
from torch.autograd import Variable
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .soft import LinearQuantActivation, linear_quant
from candle.context import Context
from candle.nested import *
from candle.proxy import *

class NAryClampFunction(ag.Function):
    def __init__(self, k=2):
        super().__init__()
        self.k = k
        self.unif_bins = np.delete(np.arange(-1, 1, 2 / (2**k - 1)), 0)
        self.unif_bins = np.delete(self.unif_bins, np.arange(self.unif_bins.shape[0] // 2))
        self.norm_bins = -erfcinv(self.unif_bins + 1) * np.sqrt(2)
        self.unif_factors = self.unif_bins * 2
        self.norm_factors = self.norm_bins * np.sqrt(np.pi / 2)
        self.factors = (self.unif_factors + self.norm_factors) / 2
        self.bins = (self.unif_bins + self.norm_bins) / 2

    def _compute_norm_bin(self, idx):
        return -erfinv(2 * idx / (2**self.k - 1) + 1) * np.sqrt(2)

    def _compute_unif_bin(self, idx):
        return 2 * idx / (2**self.k - 1)

    def forward(self, x):
        last_bin = 0
        new_x = 0 * x
        signs = x.sign()
        abs_x = x.abs()
        for b, f in zip(self.bins, self.factors):
            new_x.add_(signs * ((abs_x < b) | (last_bin <= abs_x)) * f)
            last_bin = b
        new_x.add_()

class TernaryClampFunction(ag.Function):
    @staticmethod
    def forward(ctx, x):
        delta = 0.7 * x.abs().mean()
        new_x = x / delta
        new_x = new_x.clamp(-1, 1).int().float()
        factor = x[new_x.abs().byte()]
        ctx.alpha = None
        if factor.numel() > 0:
            alpha = factor.abs().mean()
            ctx.alpha = alpha
            return alpha * new_x
        return new_x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.alpha is not None:
            return grad_output * ctx.alpha
        return grad_output

class QuantizeKFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, k):
        factor = (2**k - 1)
        return 1 / factor * (factor * x).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def dorefa_quantize_weights(x, k):
    tanh_x = x.tanh()
    return 2 * quantize_k(tanh_x / (2 * tanh_x.abs().max()) + 0.5, k) - 1

def erf_quantize(x, k):
    return 2 * quantize_k(0.5 * (1 + (x / np.sqrt(2)).tanh()), k) - 1

ternary_clamp = TernaryClampFunction.apply
quantize_k = QuantizeKFunction.apply

class TernaryWeightHook(ProxyDecorator):
    def __init__(self, layer, child, chunk=1, chunk_dim=0):
        super().__init__(layer, child)
        self.chunk = chunk
        self.chunk_dim = chunk_dim

    @property
    def sizes(self):
        return self.child.sizes

    def _chunk_apply(self, x):
        chunks = [ternary_clamp(x) for x in x.chunk(self.chunk, dim=self.chunk_dim)]
        return torch.cat(chunks, self.chunk_dim)

    def call(self, input):
        if self.chunk == 1:
            return input.apply_fn(ternary_clamp)
        return input.apply_fn(self._chunk_apply)

class DoReFaWeightHook(ProxyDecorator):
    def __init__(self, layer, child, chunk=1, chunk_dim=0, k=8, factor=False):
        super().__init__(layer, child)
        self.chunk = chunk
        self.chunk_dim = chunk_dim
        self.k = k
        self.factor = factor

    @property
    def sizes(self):
        return self.child.sizes

    def _chunk_apply(self, x):
        chunks = [_compute_dorefa(x, self.k, self.factor, weights=True) for x in x.chunk(self.chunk, dim=self.chunk_dim)]
        return torch.cat(chunks, self.chunk_dim)

    def call(self, input):
        if self.chunk == 1:
            input = input.apply_fn(lambda x: _compute_dorefa(x, self.k, self.factor, weights=True))
        else:
            input = input.apply_fn(self._chunk_apply)
        return input

def _compute_dorefa(x, k, factor, weights=False):
    if weights:
        new_x = dorefa_quantize_weights(x, k)
    else:
        new_x = quantize_k(x, k)
    if factor:
        return apply_scale(new_x, x, weights=weights) * new_x
    return new_x

def apply_scale(new_x, x, weights=False):
    if weights:
        w_t = new_x.view(-1)
        w = x.view(-1)
        factor = w_t.dot(w) / w_t.dot(w_t)
    else:
        w_t = new_x.view(new_x.size(0), -1)
        w = x.view(x.size(0), -1)
        factor = w_t.unsqueeze(-2).matmul(w.unsqueeze(-1))
        factor = factor / w_t.unsqueeze(-2).matmul(w_t.unsqueeze(-1))
        factor = factor.squeeze(1)
    return factor.expand_as(new_x) * new_x

class DoReFaActivation(nn.Module):
    def __init__(self, k, factor=False):
        super().__init__()
        self.k = k
        self.factor = factor

    def forward(self, x):
        return _compute_dorefa(x, self.k, self.factor)

class TernaryActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ternary_clamp(x)

class TernaryQuantizeContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def activation(self, k=2, min=-3, max=3, dynamic=True):
        if k > 2:
            return self.bypass(LinearQuantActivation(k, min=min, max=max, dynamic=dynamic))
        return self.bypass(TernaryActivation())

    def compose(self, layer, chunk=1, chunk_dim=0, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(TernaryWeightHook, chunk=chunk, chunk_dim=chunk_dim)
        return layer

class DoReFaQuantizeContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def activation(self, k=8):
        return self.bypass(DoReFaActivation(k))

    def compose(self, layer, chunk=1, chunk_dim=0, k=8, factor=False, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(DoReFaWeightHook, chunk=chunk, chunk_dim=chunk_dim, k=k, factor=factor)
        return layer