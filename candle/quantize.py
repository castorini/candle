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

class RestrictGradientFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, restrict_fn, apply_fn=None):
        ctx.mask_fn = restrict_fn
        ctx.apply_fn = apply_fn
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output[ctx.mask_fn(grad_output)] = 0
        if ctx.apply_fn:
            ctx.apply_fn(grad_output)
        return grad_output, None, None

def sech2(x):
    return 1 - x.tanh()**2

class MultiStepFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, scale=None, n_steps=1, limit=1, use_tanh=False):
        if use_tanh:
            ctx.save_for_backward(x, scale)
        else:
            ctx.save_for_backward(x)
        ctx.other_vars = n_steps, limit, use_tanh
        if n_steps % 2 == 0:
            return limit * ((x * (n_steps + 1) / (2 * limit)).round() * 2 / n_steps)
        else:
            return 2 * x.clamp(0, 1).ceil() - 1 # TODO: Support multistep for odd steps...

    @staticmethod
    def backward(ctx, grad_output):
        n_steps, limit, use_tanh = ctx.other_vars
        if use_tanh:
            x, scale = ctx.saved_variables
            grad_out = 0
            for idx in range(1, n_steps + 1):
                grad_out = grad_out + sech2(scale * (x + (limit - 2 * limit * idx / (n_steps + 1))))
            grad_out = scale * grad_out * grad_output
        else:
            grad_out = grad_output
        return grad_out, None, None, None, None

class SigmoidStepFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, scale=None, use_sigmoid=False):
        if use_sigmoid:
            ctx.save_for_backward(x, scale)
        else:
            ctx.save_for_backward(x)
        ctx.use_sigmoid = use_sigmoid
        return x.clamp(0, 1).ceil()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_sigmoid:
            x, scale = ctx.saved_variables
            s = (scale * x).sigmoid()
            grad_out = s * (1 - s)
            grad_out = scale * grad_out * grad_output
        else:
            grad_out = grad_output
        return grad_out

def dynamic_tanh(x, t, max_scale=50, n_steps=1, limit=1):
    x = (t <= max_scale).float() * F.tanh(t * x) + (t > max_scale).float() * multi_step(x, t, n_steps, limit, True)
    return x

def dynamic_sigmoid(x, t, max_scale=50, n_steps=1, limit=1):
    # TODO Support multistep
    x = (t <= max_scale).float() * F.sigmoid(t * x) + (t > max_scale).float() * sigmoid_step(x, t, True)

class BinaryActivation(nn.Module):
    def __init__(self, type="tanh", t=0.5, soft=True, limit=None):
        super().__init__()
        self.soft = soft
        self.use_sigmoid = type == "sigmoid"
        self.limit = limit
        if soft:
            self.fn = dynamic_tanh if type == "tanh" else dynamic_sigmoid
            if isinstance(t, nn.Parameter):
                self.scale = t
            else:
                self.scale = nn.Parameter(torch.Tensor([t]))
        else:
            self.fn = soft_sign

    def forward(self, x):
        if self.limit:
            x = restrict_grad(x, lambda x: (x < -self.limit) | (x > self.limit))
        if self.soft:
            return self.fn(x, self.scale)
        else:
            return self.fn(x, use_sigmoid=self.use_sigmoid)

def soft_sign(x, use_sigmoid=False):
    return sigmoid_step(x) if use_sigmoid else multi_step(x)

class StepQuantizeHook(ProxyDecorator):
    def __init__(self, layer, child, t=0.5, out_shape=None, soft=True, limit=1, n_steps=1, adaptive=True, out="tanh"):
        super().__init__(layer, child)
        out_shape = Package(out_shape) if out_shape else self.sizes
        self.soft = soft
        self.limit = limit
        self.clamp = (-limit, limit)
        self.n_steps = n_steps
        self.use_sigmoid = out == "sigmoid"
        if soft:
            if isinstance(t, nn.Parameter):
                self.scales = out_shape.apply_fn(lambda _: t)
                self._flattened_params = [t] if adaptive else []
            else:
                self.scales = out_shape.apply_fn(lambda _: nn.Parameter(torch.Tensor([t])))
                self._flattened_params = self.scales.reify(flat=True)

    def parameters(self):
        if self.soft:
            return self._flattened_params
        return []

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        if self.clamp:
            input.data.clamp_(*self.clamp)
        if self.soft:
            scales = self.scales.apply_fn(lambda x: restrict_grad(x, lambda y: y > 0))
            if self.use_sigmoid:
                input = input.apply_fn(lambda x, t: dynamic_sigmoid(x, t), scales)
            else:
                input = input.apply_fn(lambda x, t: dynamic_tanh(x, t, n_steps=self.n_steps, limit=self.limit), scales)
        else:
            input = input.apply_fn(lambda x: soft_sign(x, use_sigmoid=self.use_sigmoid))
        return input

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
restrict_grad = RestrictGradientFunction.apply
multi_step = MultiStepFunction.apply
sigmoid_step = SigmoidStepFunction.apply

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

class StepQuantizeContext(Context):
    def __init__(self, soft=True, **kwargs):
        super().__init__(**kwargs)
        self.soft = soft

    def list_scale_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, StepQuantizeHook))
        return super().list_params(lambda proxy: isinstance(proxy, StepQuantizeHook))

    def list_model_params(self):
        return self.list_scale_params(True)

    def quantize_loss(self, lambd):
        loss = 0
        for param in self.list_model_params():
            loss = loss + lambd / (param.norm(p=0.66) + 1E-8)
        return loss

    def activation(self, type="tanh", scale=0.5, soft=True, limit=1):
        return self.bypass(BinaryActivation(type, scale, soft, limit=limit))

    def compose(self, layer, soft=True, limit=1, scale=0.5, adaptive=True, n_steps=1, out="tanh", **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(StepQuantizeHook, soft=soft, limit=limit, t=scale, n_steps=n_steps, 
            adaptive=adaptive, out=out)
        return layer
