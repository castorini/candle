import math

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import Context
from .nested import *
from .proxy import *

class WeightMaskGroup(ProxyDecorator):
    def __init__(self, child, init_value=1, stochastic=False):
        super().__init__(child)
        self.masks = self.build_masks(init_value)
        self.stochastic = stochastic
        self._flattened_masks = self.masks.reify(flat=True)

    def expand_masks(self):
        raise NotImplementedError

    def build_masks(self, init_value):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError

    def parameters(self):
        return self._flattened_masks

    def print_info(self):
        super().print_info()
        new_size = int(self._flattened_masks[0].sum().cpu().data[0])
        print("{}: {} => {}".format(type(self), self.child.sizes, new_size))

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        masks = self.expand_masks()
        return input * masks

class Channel2DMask(WeightMaskGroup):
    def __init__(self, child, **kwargs):
        super().__init__(child, **kwargs)

    def build_masks(self, init_value):
        return Package([nn.Parameter(init_value * torch.ones(self.child.sizes[0][0]))])

    def split(self, root):
        param = root.parameters()[0]
        split_root = param.view(param.size(0), -1).permute(1, 0)
        return Package([split_root])

    def expand_masks(self):
        mask = self._flattened_masks[0]
        if self.stochastic:
            mask = mask.clamp(0, 1).bernoulli()
        sizes = self.child.sizes[0]
        expand_weight = mask.expand(sizes[3], sizes[2], sizes[1], -1).permute(3, 2, 1, 0)
        expand_bias = mask
        return Package([expand_weight, expand_bias])

class LinearRowMask(WeightMaskGroup):
    def __init__(self, child, **kwargs):
        super().__init__(child, **kwargs)

    def build_masks(self, init_value):
        return Package([nn.Parameter(init_value * torch.ones(self.child.sizes[0][0]))])

    def split(self, root):
        return Package([root.parameters()[0].permute(1, 0)])

    def expand_masks(self):
        mask = self._flattened_masks[0]
        if self.stochastic:
            mask = mask.clamp(0, 1).bernoulli()
        expand_weight = mask.expand(self.child.sizes[0][1], -1).permute(1, 0)
        expand_bias = mask
        return Package([expand_weight, expand_bias])

class LinearColMask(WeightMaskGroup):
    def __init__(self, child, **kwargs):
        super().__init__(child, **kwargs)
        self._dummy = nn.Parameter(torch.ones(child.sizes[1][0]))
        self._flattened_masks.append(self._dummy)

    def build_masks(self, init_value):
        return Package([nn.Parameter(init_value * torch.ones(self.child.sizes[0][1]))])

    def split(self, root):
        return Package([root.parameters()[0]])

    def expand_masks(self):
        mask = self._flattened_masks[0]
        if self.stochastic:
            mask = mask.clamp(0, 1).bernoulli()
        expand_weight = mask.expand(self.child.sizes[0][0], -1)
        expand_bias = self._dummy
        return Package([expand_weight, expand_bias])

class WeightMask(ProxyDecorator):
    def __init__(self, child, init_value=1, stochastic=False):
        super().__init__(child)
        def create_mask(size):
            return nn.Parameter(torch.ones(*size) * init_value)
        self.masks = child.package.size().apply_fn(create_mask)
        self._flattened_masks = self.masks.reify(flat=True)
        self.stochastic = stochastic

    def parameters(self):
        return self._flattened_masks

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, input):
        if self.stochastic:
            return input * self.masks.clamp(0, 1).bernoulli()
        return input * self.masks

def _group_rank_norm(context, proxies, p=1):
    return [proxy.split(proxy.root).norm(p, 0) for proxy in proxies]

def _group_rank_l1(context, proxies):
    return _group_rank_norm(context, proxies, p=1)

def _group_rank_l2(context, proxies):
    return _group_rank_norm(context, proxies, p=2)

def _single_rank_magnitude(context, proxies):
    providers = context.list_providers()
    ranks = [provider.package.abs() for provider in providers]
    return ranks

_single_rank_methods = dict(magnitude=_single_rank_magnitude)
_group_rank_methods = dict(l1_norm=_group_rank_l1, l2_norm=_group_rank_l2)

class PruneContext(Context):
    def __init__(self, stochastic=False, **kwargs):
        super().__init__(**kwargs)
        self.stochastic = stochastic

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        if kwargs.get("active"):
            layer.hook_weight(WeightMask, stochastic=self.stochastic)
        return layer

    def list_mask_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, WeightMask))
        return super().list_params(lambda proxy: isinstance(proxy, WeightMask))

    def list_model_params(self):
        return self.list_mask_params(inverse=True)

    def count_unpruned(self):
        return sum(p.sum().cpu().data[0] for p in self.list_mask_params())

    def clip_all_masks(self):
        for p in self.list_mask_params():
            p.data.clamp_(0, 1)

    def prune(self, percentage, method="magnitude", method_map=_single_rank_methods, mask_type=WeightMask):
        rank_call = method_map[method]
        proxies = self.list_proxies("weight_hook", mask_type)
        weights_list = rank_call(self, proxies)
        for weights, proxy in zip(weights_list, proxies):
            for weight, mask in flatten_zip(weights.reify(), proxy.masks.reify()):
                _, indices = torch.sort(weight.view(-1))
                ne0_indices = indices[mask.view(-1)[indices] != 0]
                if ne0_indices.size(0) <= 1:
                    continue
                length = math.ceil(ne0_indices.size(0) * percentage / 100)
                indices = ne0_indices[:length]
                if indices.size(0) > 0:
                    mask.data.view(-1)[indices.data] = 0

class GroupPruneContext(PruneContext):
    def __init__(self, stochastic=False, **kwargs):
        super().__init__(**kwargs)
        self.stochastic = stochastic

    def compose(self, layer, **kwargs):
        layer = super().compose(layer, **kwargs)
        layer.hook_weight(self.find_mask_type(type(layer), kwargs.get("prune", "out")), stochastic=self.stochastic)
        return layer

    def find_mask_type(self, layer_type, prune="out"):
        if layer_type == ProxyLinear and prune == "out":
            return LinearRowMask
        elif layer_type == ProxyLinear and prune == "in":
            return LinearColMask
        elif layer_type == ProxyConv2d  and prune == "out":
            return Channel2DMask
        else:
            raise ValueError("Layer type unsupported!")

    def list_mask_params(self, inverse=False):
        if inverse:
            return super().list_params(lambda proxy: not isinstance(proxy, WeightMaskGroup))
        return super().list_params(lambda proxy: isinstance(proxy, WeightMaskGroup))

    def count_unpruned(self):
        group_masks = self.list_proxies("weight_hook", WeightMaskGroup)
        return sum(sum(m.expand_masks().sum().cpu().data[0].reify(flat=True)) for m in group_masks)

    def prune(self, percentage, method="l2_norm", method_map=_group_rank_methods, mask_type=WeightMaskGroup):
        super().prune(percentage, method, method_map, mask_type)
