from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .proxy import Proxy, ProxyDecorator, nested_map, flatten, flatten_zip

class WeightMask(ProxyDecorator):
    def __init__(self, child, init_value=1):
        super().__init__(child)
        self.masks = nested_map(lambda size: nn.Parameter(torch.ones(*size) * init_value), child.sizes)
        self._flattened_masks = flatten(self.masks)

    def parameters(self):
        return self._flattened_masks

    @property
    def sizes(self):
        return self.child.sizes

    def apply(self, weights, masks):
        applied_masks = []
        for w, m in zip(weights, masks):
            if isinstance(w, list):
                applied_masks.append(self.apply(w, m))
            else:
                applied_masks.append(w * m)
        return applied_masks

    def call(self, *weights):
        return self.apply(weights, self.masks)

class StochasticWeightMask(WeightMask):
    def __init__(self, child, init_prob=1):
        super().__init__(child, init_value=init_prob)

    def call(self, *weights):
        return self.apply(weights, nested_map(lambda m: torch.bernoulli(torch.clamp(m, 0, 1)), self.masks))

def _rank_magnitude(context, proxies):
    providers = context.list_providers()
    ranks = [nested_map(lambda p: torch.abs(p), provider.raw_parameters) for provider in providers]
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
            for weight, mask in flatten_zip(weights, proxy.masks):
                _, indices = torch.sort(weight.view(-1))
                ne0_indices = indices[mask.view(-1)[indices] != 0]
                if ne0_indices.size(0) == 0:
                    continue
                length = int(ne0_indices.size(0) * percentage / 100)
                indices = ne0_indices[:length]
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0

_rank_methods = dict(magnitude=_rank_magnitude)
