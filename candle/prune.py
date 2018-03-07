from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .nested import *
from .proxy import Proxy, ProxyDecorator

class WeightMask(ProxyDecorator):
    def __init__(self, child, init_value=1):
        super().__init__(child)
        self.masks = Package(nested_map(lambda size: nn.Parameter(torch.ones(*size) * init_value), child.sizes))
        self._flattened_masks = self.masks.reify(flat=True)

    def parameters(self):
        return self._flattened_masks

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        return input * self.masks

class StochasticWeightMask(WeightMask):
    def __init__(self, child, init_prob=1):
        super().__init__(child, init_value=init_prob)

    def call(self, input):
        return input * self.masks.clamp(0, 1).bernoulli()

def _rank_magnitude(context, proxies):
    providers = context.list_providers()
    ranks = [provider.package.abs() for provider in providers]
    return ranks

class PruneContext(Context):
    def __init__(self, stochastic=False, **kwargs):
        super().__init__(**kwargs)
        self.stochastic = stochastic

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        hook = StochasticWeightMask if self.stochastic else WeightMask
        layer.hook_weight(hook)
        return layer

    def list_mask_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, WeightMask))
        return super().list_params(lambda proxy: isinstance(proxy, WeightMask))

    def count_unpruned(self):
        return sum(p.sum().cpu().data[0] for p in self.list_mask_params())

    def clip_all_masks(self):
        for p in self.list_mask_params():
            p.data.clamp_(0, 1)

    def prune(self, percentage, method="magnitude"):
        rank_call = _rank_methods[method]
        proxies = self.list_proxies("weight_hook", WeightMask)
        weights_list = rank_call(self, proxies)
        for weights, proxy in zip(weights_list, proxies):
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()):
                _, indices = torch.sort(weight.view(-1))
                ne0_indices = indices[mask.view(-1)[indices] != 0]
                if ne0_indices.size(0) == 0:
                    continue
                length = int(ne0_indices.size(0) * percentage / 100)
                indices = ne0_indices[:length]
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0

_rank_methods = dict(magnitude=_rank_magnitude)
