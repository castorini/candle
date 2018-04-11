import gc
import copy
import math

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

# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/fake_quant_ops_functor.h
class LinearQuantFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, bits, min_, max_, dynamic=True):
        if dynamic:
            max_ = max(np.abs(x.max().item()), np.abs(x.min().item()))
            min_ = -max_
        range = max_ - min_
        ctx.skip = False
        if range == 0:
            ctx.skip = True
            return x
        ctx.save_for_backward(x)
        quant_max = (1 << bits) - 1
        scale = range / quant_max
        zero_point = -min_ / scale
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
        if ctx.skip:
            return grad_output, None, None, None, None
        x, = ctx.saved_variables
        nudged_min, nudged_max = ctx.values
        grad_output[(x < nudged_min) | (x > nudged_max)] = 0
        return grad_output, None, None, None, None

class LinearQuantActivation(nn.Module):
    def __init__(self, bits, min=-3, max=3, dynamic=True):
        super().__init__()
        self.args = (bits, min, max, dynamic)

    def forward(self, x):
        return linear_quant(x, *self.args)

def sech2(x):
    return 1 - x.tanh()**2

class LeakySigmoidFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, alpha=1E-1):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        return x.sigmoid()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        s = x.sigmoid()
        grad_out = grad_output * (s * (1 - s) + ctx.alpha)
        return grad_out, None

class LeakyTanhFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, alpha=1E-1):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        return x.tanh()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        grad_out = grad_output * (sech2(x) + ctx.alpha)
        return grad_out, None

class BinaryActivation(nn.Module):
    def __init__(self, t=0., soft=True, stochastic=False):
        super().__init__()
        self.soft = soft
        self.scale = t
        self.stochastic = stochastic

    def forward(self, x):
        if self.soft:
            return binary_tanh(x, self.scale, stochastic=self.stochastic)
        else:
            return binary_tanh(x, stochastic=self.stochastic)

class StepQuantizeHook(ProxyDecorator):
    def __init__(self, layer, child, t=0., out_shape=None, soft=True, limit=1, init_uniform=True, rescale=False):
        super().__init__(layer, child)
        out_shape = Package(out_shape) if out_shape else self.sizes
        self.soft = soft
        self.limit = limit
        self.clamp = (-limit, limit)
        self.scale = t
        self.rescale = rescale

        if init_uniform:
            self.layer.init_weights(lambda x: x.uniform_(-1, 1))

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        from .dorefa import apply_scale
        if self.clamp:
            input.data.clamp_(*self.clamp)
        if self.soft:
            output = input.apply_fn(lambda x: binary_tanh(x, self.scale))
        else:
            output = input.apply_fn(binary_tanh)
        if self.rescale:
            output = output.apply_fn(lambda x, old_x: apply_scale(x, old_x, True), output)
        return output

class QuantizeHook(ProxyDecorator):
    def __init__(self, layer, child, bits=8, min=-3, max=3):
        super().__init__(layer, child)
        self.args = (bits, min, max)

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        input = input.apply_fn(lambda x: linear_quant(x, *self.args))
        return input

def hard_sigmoid(x):
    return inclusive_clamp(((x + 1) / 2), 0, 1)

def binary_tanh(x, scale=1, stochastic=False):
    if stochastic:
        return 2 * soft_bernoulli(x.sigmoid(), scale) - 1
    else:
        return 2 * soft_round(hard_sigmoid(x), scale) - 1

class SoftRoundFunction(ag.Function):
    @staticmethod
    def forward(ctx, input, scale=1):
        if scale < 1:
            return scale * input.round() + (1 - scale) * input
        else:
            return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SoftBernoulliFunction(ag.Function):
    @staticmethod
    def forward(ctx, alpha, scale=1):
        if scale < 1:
            return scale * alpha.bernoulli() + (1 - scale) * alpha
        else:
            return alpha.bernoulli()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class InclusiveClamp(ag.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x)
        ctx.limit = (a, b)
        return x.clamp(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        a, b = ctx.limit
        grad_output[(x < a) | (x > b)] = 0
        return grad_output, None, None

def logb2(x):
    return torch.log(x) / np.log(2)

def ap2(x):
    x = x.sign() * torch.pow(2, torch.round(logb2(x.abs())))
    return x

class ApproxPow2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ap2(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        return grad_output

class PassThroughDivideFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_variables
        return grad_output, -grad_output * x / y**2

class QuantizedBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.125, eps=1e-5, affine=True, k=8, min=-2, max=2):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.k = k
        self.min = min
        self.max = max
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.reset_parameters()
        self._init_quantize_fn()

    def _init_quantize_fn(self):
        if self.k == 1:
            self.quantize_fn = approx_pow2
        else:
            self.quantize_fn = lambda x: linear_quant(x, self.k, self.min, self.max)

    def reset_parameters(self):
        self.mean.zero_()
        self.var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.weight.data -= 0.5
            self.bias.data.zero_()

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
            self.mean = self.quantize_fn((1 - self.momentum) * self.mean + self.momentum * new_mean)
        mean = self._convert_param(input, self.mean)
        ctr_in = self.quantize_fn(input - Variable(mean))

        if self.training:
            new_var = self._reorg(ctr_in * ctr_in).mean(1).data
            self.var = self.quantize_fn((1 - self.momentum) * self.var + self.momentum * new_var)
        var = self._convert_param(input, self.var)
        x = self.quantize_fn(ctr_in / self.quantize_fn(torch.sqrt(Variable(var) + self.eps)))

        if self.affine:
            w1 = self._convert_param(x, self.weight)
            b1 = self._convert_param(x, self.bias)
            y = self.quantize_fn(self.quantize_fn(w1) * x + self.quantize_fn(b1))
        else:
            y = x
        return y

approx_pow2 = ApproxPow2Function.apply
passthrough_div = PassThroughDivideFunction.apply
soft_round = SoftRoundFunction.apply
linear_quant = LinearQuantFunction.apply
leaky_tanh = LeakyTanhFunction.apply
leaky_sigmoid = LeakySigmoidFunction.apply
inclusive_clamp = InclusiveClamp.apply
soft_bernoulli = SoftBernoulliFunction.apply

class QuantizeContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(QuantizeHook)
        if kwargs.get("output", True):
            layer.hook_output(QuantizeHook)
        return layer

class StepQuantizeContext(Context):
    def __init__(self, scale_alpha=0, soft=True, **kwargs):
        super().__init__(**kwargs)
        self.soft = soft
        self.scale_alpha = scale_alpha

    def list_scale_params(self, inverse=False):
        check = lambda x: isinstance(x, StepQuantizeHook) or isinstance(x.layer, BinaryActivation)
        if inverse:
            return super().list_params(lambda proxy: not check(proxy))
        return super().list_params(check)

    def scale_delta(self, scale):
        deltas = []
        n_elems = []
        for layer in self.layers:
            weights = layer.weight_provider().reify(flat=True)
            for weight in weights:
                weight = weight.view(-1).abs()
                weight = weight[weight > scale]
                n = weight.numel()
                try:
                    idx = int(n * self.scale_alpha)
                    if idx == 0:
                        continue
                    delta = torch.sort(weight)[0][idx] - scale
                    deltas.append(delta.cpu().data.item())
                    n_elems.append(float(idx + 1))
                except:
                    continue

        tot_elem = sum(n_elems)
        n_elems = Package(n_elems)
        deltas = Package(deltas)
        return sum((n_elems * deltas).reify(flat=True)) / tot_elem

    def list_model_params(self):
        return self.list_scale_params(True)

    def quantize_loss(self, lambd):
        loss = 0
        for param in self.list_model_params():
            loss = loss + lambd / (param.norm(p=0.66) + 1E-8)
        return loss

    def activation(self, type="tanh", scale=0., soft=True, stochastic=False):
        return self.bypass(BinaryActivation(scale, soft, stochastic=stochastic))

    def compose(self, layer, soft=True, limit=1, scale=0., **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(StepQuantizeHook, soft=soft, limit=limit, t=scale)
        return layer
