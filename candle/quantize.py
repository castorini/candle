import copy

from torch.autograd import Variable
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .estimator import *
from .nested import *
from .proxy import *

# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/fake_quant_ops_functor.h
class LinearQuantFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, bits, min, max):
        range = max - min
        ctx.save_for_backward(x)
        quant_max = (1 << bits) - 1
        scale = range / quant_max
        zero_point = -min / scale
        if zero_point < 0:
            nudged_zero_point = 0
        elif zero_point > quant_max:
            nudged_zero_point = quant_max
        else:
            nudged_zero_point = round(zero_point)
        
        nudged_min = -nudged_zero_point * scale
        nudged_max = (quant_max - nudged_zero_point) * scale
        ctx.values = nudged_min, nudged_max

        clamped_shifted = x.clamp(nudged_min, nudged_max) - nudged_min
        quantized = (clamped_shifted / scale + 0.5).floor() * scale + nudged_min
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        nudged_min, nudged_max = ctx.values
        grad_output[(x < nudged_min) | (x > nudged_max)] = 0
        return grad_output, None, None, None

class QuantizeHook(ProxyDecorator):
    def __init__(self, layer, child, bits=8, min=-6, max=6):
        super().__init__(layer, child)
        self.args = (bits, min, max)

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        if isinstance(input, Package):
            return input.apply_fn(lambda x: linear_quant(x, *self.args))
        return linear_quant(input, *self.args)

class BinaryTanhFunction(ag.Function):
    @staticmethod
    def forward(ctx, input):
        return 2 * input.clamp(0, 1).ceil() - 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class HardRoundFunction(ag.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ApproxPow2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ap2(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        return grad_output

class HardDivideFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return (x / y).floor()

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_variables
        return grad_output / y, -grad_output * x / y**2

class BinaryBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.frozen = False
        if self.affine:
            self.weight1 = nn.Parameter(torch.Tensor(num_features))
            self.bias1 = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.mean.zero_()
        self.var.fill_(1)
        if self.affine:
            self.weight1.data.uniform_()
            self.weight1.data -= 0.5
            self.bias1.data.zero_()

    def _convert_param(self, x, param):
        param = param.repeat(x.size(0), 1)
        for _ in x.size()[2:]:
            param = param.unsqueeze(-1)
        param = param.expand(-1, -1, *x.size()[2:])
        return param

    def _reorg(self, input):
        axes = [1, 0]
        axes.extend(range(2, input.dim()))
        return input.permute(*axes).contiguous().view(input.size(1), -1)

    def forward(self, input):
        if self.training:
            new_mean = self._reorg(input).mean(1).data
            self.mean = (1 - self.momentum) * self.mean + self.momentum * new_mean
        mean = self._convert_param(input, self.mean.round())
        ctr_in = input - Variable(mean)

        if self.training:
            new_var = self._reorg(ctr_in * approx_pow2(ctr_in)).mean(1).data
            self.var = (1 - self.momentum) * self.var + self.momentum * new_var
        var = self._convert_param(input, self.var)
        x = ctr_in * approx_pow2(1 / torch.sqrt(Variable(var) + self.eps))

        if self.affine:
            w1 = self._convert_param(x, self.weight1)
            b1 = self._convert_param(x, self.bias1)
            y = approx_pow2(w1) * x + approx_pow2(b1)
        else:
            y = x
        return y

approx_pow2 = ApproxPow2Function.apply
binary_tanh = BinaryTanhFunction.apply
hard_divide = HardDivideFunction.apply
hard_round = HardRoundFunction.apply
linear_quant = LinearQuantFunction.apply

class BinaryTanh(nn.Module):
    def forward(self, x):
        return binary_tanh(x)

class QuantizeContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(QuantizeHook)
        if kwargs.get("output", True):
            layer.hook_output(QuantizeHook)
        return layer
