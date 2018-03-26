import enum

from torch.autograd import Variable
from scipy.optimize import fsolve
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import *
from .estimator import Function
from .nested import *
from .proxy import *

class RNNWeightDrop(ProxyDecorator):
    def __init__(self, layer, child, layer_indices=[], mode="hidden", p=0.3):
        super().__init__(layer, child)
        self.mode = mode
        self.p = p
        self.layer_indices = set(layer_indices)

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        def apply_dropout(layer_weights):
            if self.mode == "hidden":
                layer_weights[1] = F.dropout(layer_weights[1], training=self.layer.training, p=self.p)
            elif self.mode == "input":
                layer_weights[0] = F.dropout(layer_weights[0], training=self.layer.training, p=self.p)
            else:
                raise ValueError(f"{self.mode} not supported!")
        weights = input.reify()
        for i, x in enumerate(weights):
            if self.layer_indices and i not in self.layer_indices:
                continue
            apply_dropout(x)
        return Package.reshape_into(input.nested_shape, flatten(weights))

def solve_log_cosh(scale=1, limit=1):
    def loss(x):
        a, b = x
        f1 = a * np.log(np.cosh(b * limit)) - scale * limit**2
        f2 = a * np.log(np.cosh(b * limit / 2)) - scale * limit**2 / 4
        return f1, f2
    a, b = fsolve(loss, (scale * limit * 10, scale * limit / 10), maxfev=5000)
    return a, b
