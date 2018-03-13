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

class BernoulliHook(ProxyDecorator):
    def __init__(self, layer, child):
        super().__init__(layer, child)
        def normalize_weights(weight):
            weight.data -= weight.data.mean()
        self.frozen = False
        child.package.iter_fn(normalize_weights)
        self._distribution = SoftBernoulliDistribution(child.package)

    def distribution(self):
        return self._distribution

    @property
    def sizes(self):
        return self.child.sizes

    def freeze(self, x):
        self.frozen_x = x
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
        self.frozen_x = None

    def call(self, input):
        if self.frozen:
            return 2 * self.frozen_x - 1
        return 2 * self.distribution().draw() - 1

class RoundHook(ProxyDecorator):
    def __init__(self, layer, child):
        super().__init__(layer, child)

    def call(self, input):
        return input.apply_fn(lambda x, size: hard_round(x * np.prod(size, dtype=float)), input.size()) \
            if isinstance(input, Package) else hard_round(input * np.prod(input.size(), dtype=float))

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

class BinaryTanh(nn.Module):
    def forward(self, x):
        return binary_tanh(x)

class BinaryContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(RoundHook)
        if kwargs.get("round", True):
            layer.hook_output(RoundHook)
        return layer
