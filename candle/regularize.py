import enum

from torch.autograd import Variable
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

