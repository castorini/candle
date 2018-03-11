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
    def __init__(self, child):
        super().__init__(child)
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
            return self.frozen_x
        return 2 * self.distribution.draw() - 1

class RoundHook(ProxyDecorator):
    def __init__(self, child):
        super().__init__(child)

    def call(self, input):
        return input.round()

class BinaryTanhFunction(ag.Function):
    @staticmethod
    def forward(ctx, input):
        return 2 * input.clamp(0, 1).ceil() - 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class IntegralBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.frozen = False
        if self.affine:
            w1 = self.weight1 = nn.Parameter(torch.Tensor(num_features))
            w2 = self.weight2 = nn.Parameter(torch.Tensor(num_features))
            b1 = self.bias1 = nn.Parameter(torch.Tensor(num_features))
            b2 = self.bias2 = nn.Parameter(torch.Tensor(num_features))
            b3 = self.bias3 = nn.Parameter(torch.Tensor(num_features))
            self._distribution = SoftBernoulliDistribution(Package([w1, w2, b1, b2, b3]))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self._distribution = None
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.reset_parameters()

    def distribution(self):
        return self._distribution

    def freeze(self, x):
        self.frozen_x = x
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
        self.frozen_x = None

    def reset_parameters(self):
        self.mean.zero_()
        self.var.fill_(1)
        if self.affine:
            self.weight1.data.uniform_()
            self.weight1.data -= 0.5
            self.bias1.data.zero_()
            self.weight2.data.uniform_()
            self.weight2.data -= 0.5
            self.bias2.data.zero_()
            self.bias3.data.zero_()

    def _convert_param(self, x, param, random=False):
        if random:
            param = param.sigmoid().bernoulli() if self.training else param.sigmoid().round()
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
            new_var = self._reorg(ctr_in * ctr_in).mean(1).data
            self.var = (1 - self.momentum) * self.var + self.momentum * new_var
        var = self._convert_param(input, self.var)
        div = torch.sqrt(Variable(var) + self.eps).round()
        div[div == 0] = 1
        x = (ctr_in / div).floor()

        if self.affine:
            w1 = self._convert_param(x, self.frozen_x[0] if self.frozen else self.weight1, random=True)
            w2 = self._convert_param(x, self.frozen_x[1] if self.frozen else self.weight2, random=True)
            b1 = self._convert_param(x, self.frozen_x[2] if self.frozen else self.bias1, random=True)
            b2 = self._convert_param(x, self.frozen_x[3] if self.frozen else self.bias2, random=True)
            y = (2 * w1 + 2 * w2 - 2) * x + 2 * b1 + 2 * b2 - 2
        else:
            y = x
        return y.singleton

binary_tanh = BinaryTanhFunction.apply

class BinaryTanh(nn.Module):
    def forward(self, x):
        return binary_tanh(x)

class BernoulliContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dists = []
        self.params = []

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        self.dists.append(layer.hook_weight(BernoulliHook))
        self.params.append(list(layer.parameters()))
        if kwargs.get("round"):
            layer.hook_output(RoundHook)
        return layer

    def clip_all_masks(self):
        for p in self.list_mask_params():
            p.data.clamp_(0, 1)

    def wrap(self, layer, **kwargs):
        if isinstance(layer, IntegralBatchNorm):
            self.bypass(layer)
            if layer.distribution():
                self.dists.append(layer)
                self.params.append(list(layer.parameters()))
            return layer
        else:
            return super().wrap(layer, **kwargs)

    @property
    def parameters(self):
        return self.cache("_params", lambda: Package(self.params))

    @property
    def distributions(self):
        return self.cache("_dists", lambda: Package(self.dists))

    def estimate_gradient(self, loss_fn):
        def f(b):
            self.distributions.iter_fn(lambda layer, params: layer.freeze(params), b)
            return loss_fn()
        estimator = REINFORCEEstimator(f, self.distributions.distribution())
        return estimator.estimate_gradient(self.parameters)

