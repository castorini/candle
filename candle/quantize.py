from torch.autograd import Variable
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .functional import *
from .model import SerializableModule

def huber_transform(x):
    x = x.clone()
    abs_x = torch.abs(x)
    x[abs_x < 1] = x[abs_x < 1].pow_(2).mul_(0.5)
    x[abs_x >= 1] = x[abs_x >= 1].abs_()
    return x

def quantized_loss(params, const):
    loss = 0
    for p in params:
        loss = loss + const * (torch.abs(p) - 1).norm(p=1)
    return loss

def logb2(x):
    return torch.log(x) / np.log(2)

def ap2(x):
    x = x.sign() * torch.pow(2, torch.round(logb2(x.abs())))
    return x

# Adapted from PyTorch code
class ShiftBatchNorm(SerializableModule):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
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
        new_mean = self._reorg(input).mean(1).data
        self.mean = (1 - self.momentum) * self.mean + self.momentum * new_mean
        mean = self._convert_param(input, self.mean)
        ctr_in = input - Variable(mean)

        new_var = self._reorg(ctr_in * approx_pow2(ctr_in)).mean(1).data
        self.var = (1 - self.momentum) * self.var + self.momentum * new_var
        var = self._convert_param(input, self.var)
        x = ctr_in * approx_pow2(1 / torch.sqrt(Variable(var) + self.eps))

        if self.affine:
            w = self._convert_param(x, self.weight)
            b = self._convert_param(x, self.bias)
            y = approx_pow2(w) * x + approx_pow2(b)
        else:
            y = x
        return y
