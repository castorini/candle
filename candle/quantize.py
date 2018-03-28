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
            grad_output = ctx.apply_fn(grad_output)
        return grad_output, None, None

def sech2(x):
    return 1 - x.tanh()**2

class TanhStepFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, scale=None, soft_backward=False, alpha=0):
        if soft_backward:
            ctx.save_for_backward(x, scale)
        else:
            ctx.save_for_backward(x)
        ctx.soft_backward = soft_backward
        ctx.alpha = alpha
        return 2 * x.clamp(0, 1).ceil() - 1

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.soft_backward:
            x, scale = ctx.saved_variables
            grad_out = (scale * sech2(scale * x) + ctx.alpha) * grad_output
        else:
            grad_out = grad_output
        return grad_out, None, None, None

class ReLUStepFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, alpha=0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return 2 * x.clamp(0, 1).ceil() - 1

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_variables
        grad_out = (x > 0).float() * grad_output + (x <= 0).float() * ctx.alpha
        return grad_out, None

class SigmoidStepFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, scale=None, soft_backward=False, alpha=0):
        if soft_backward:
            ctx.save_for_backward(x, scale)
        else:
            ctx.save_for_backward(x)
        ctx.soft_backward = soft_backward
        ctx.alpha = alpha
        return x.clamp(0, 1).ceil()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.soft_backward:
            x, scale = ctx.saved_variables
            s = (scale * x).sigmoid()
            grad_out = s * (1 - s)
            grad_out = (scale * grad_out + ctx.alpha) * grad_output
        else:
            grad_out = grad_output
        return grad_out, None, None, None

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

def dynamic_relu(x, t, max_scale=50, alpha=1E-2):
    return relu_step(x, alpha)

def dynamic_tanh(x, t, max_scale=50, alpha=5E-1):
    if t.data[0] <= max_scale:
        x = leaky_tanh(t * x, alpha)
    else:
        x = tanh_step(x, t, True, alpha)
    return x

def dynamic_sigmoid(x, t, max_scale=50, alpha=1E-1):
    if t.data[0] <= max_scale:
        x = leaky_sigmoid(t * x, alpha)
    else:
        x = sigmoid_step(x, t, True, alpha)
    return x

class BinaryActivation(nn.Module):
    def __init__(self, type="tanh", t=0.5, soft=True, limit=None, scale_lr=1):
        super().__init__()
        self.soft = soft
        self.use_sigmoid = type == "sigmoid"
        self.limit = limit
        self.lr_fn = None
        if scale_lr != 1:
            self.lr_fn = lambda x: scale_lr * x
        if soft:
            self.fn = dynamic_tanh if type == "tanh" else dynamic_sigmoid
            if isinstance(t, nn.Parameter):
                self.scale = t
            else:
                self.scale = nn.Parameter(torch.Tensor([t]))
        else:
            self.fn = soft_sign

    def forward(self, x):
        if self.limit is not None:
            x = restrict_grad(x, lambda x: (x < -self.limit) | (x > self.limit))
        if self.soft:
            scale = restrict_grad(self.scale, lambda y: y > 0, self.lr_fn)
            return self.fn(x, scale)
        else:
            return self.fn(x, use_sigmoid=self.use_sigmoid)

def soft_sign(x, use_sigmoid=False):
    return sigmoid_step(x) if use_sigmoid else tanh_step(x)

class StepQuantizeHook(ProxyDecorator):
    def __init__(self, layer, child, t=0.5, out_shape=None, soft=True, limit=1, adaptive=True, scale_lr=1, out="tanh"):
        super().__init__(layer, child)
        out_shape = Package(out_shape) if out_shape else self.sizes
        self.soft = soft
        self.limit = limit
        self.clamp = (-limit, limit)
        self.use_sigmoid = out == "sigmoid"
        self.use_tanh = out == "tanh"
        self.use_relu = out == "relu"
        self.lr_fn = None
        if scale_lr != 1:
            self.lr_fn = lambda x: scale_lr * x
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
            scales = self.scales.apply_fn(lambda x: restrict_grad(x, lambda y: y > 0, self.lr_fn))
            if self.use_relu:
                input = input.apply_fn(dynamic_relu, scales)
            elif self.use_sigmoid:
                input = input.apply_fn(dynamic_sigmoid, scales)
            elif self.use_tanh:
                input = input.apply_fn(dynamic_tanh, scales)
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

class BinaryBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.125, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
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
            self.mean = (1 - self.momentum) * self.mean + self.momentum * new_mean
        mean = self._convert_param(input, self.mean)
        ctr_in = input - Variable(mean)
        ctr_in_ap2 = approx_pow2(ctr_in)

        if self.training:
            new_var = self._reorg(ctr_in * ctr_in_ap2).mean(1).data
            self.var = (1 - self.momentum) * self.var + self.momentum * new_var
        var = self._convert_param(input, self.var)
        x = ctr_in_ap2 * approx_pow2(1 / torch.sqrt(Variable(var) + self.eps))
        x = approx_pow2(x)

        if self.affine:
            w1 = self._convert_param(x, self.weight)
            b1 = self._convert_param(x, self.bias)
            y = approx_pow2(w1) * x + b1
        else:
            y = x
        return y

approx_pow2 = ApproxPow2Function.apply
binary_tanh = BinaryTanhFunction.apply
passthrough_div = PassThroughDivideFunction.apply
hard_round = HardRoundFunction.apply
linear_quant = LinearQuantFunction.apply
restrict_grad = RestrictGradientFunction.apply
tanh_step = TanhStepFunction.apply
sigmoid_step = SigmoidStepFunction.apply
relu_step = ReLUStepFunction.apply
leaky_tanh = LeakyTanhFunction.apply
leaky_sigmoid = LeakySigmoidFunction.apply

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
        check = lambda x: isinstance(x, StepQuantizeHook) or isinstance(x.layer, BinaryActivation)
        if inverse:
            return super().list_params(lambda proxy: not check(proxy))
        return super().list_params(check)

    def list_model_params(self):
        return self.list_scale_params(True)

    def quantize_loss(self, lambd):
        loss = 0
        for param in self.list_model_params():
            loss = loss + lambd / (param.norm(p=0.66) + 1E-8)
        return loss

    def activation(self, type="tanh", scale=0.5, soft=True, limit=1, scale_lr=1):
        return self.bypass(BinaryActivation(type, scale, soft, limit=limit, scale_lr=scale_lr))

    def compose(self, layer, soft=True, limit=1, scale=0.5, scale_lr=1, adaptive=True, out="tanh", **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = layer.hook_weight(StepQuantizeHook, soft=soft, limit=limit, t=scale,
            adaptive=adaptive, scale_lr=scale_lr, out=out)
        return layer

# Adapted from pytorch repo
class SignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                d_p = d_p.sign()
                p.data.add_(-group['lr'], d_p)

        return loss