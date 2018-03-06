from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .nested_list import *
from .proxy import Proxy, ProxyDecorator

class StochasticBinaryWeightHook(ProxyDecorator):
    def __init__(self, child):
        super().__init__(child)

    def parameters(self):
        return []

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, *weights):
        return nested_map(lambda w: torch.bernoulli(w.clamp(0, 1)), weights)

class BinarizedContext(Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        layer.hook_weight(StochasticBinaryWeightHook)
        return layer

