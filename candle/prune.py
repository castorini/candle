from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import SerializableModule

def prune_magnitude(weights, weight_masks, **kwargs):
    percentage = kwargs.get("percentage", 50)
    for weight, weight_mask in zip(weights, weight_masks):
        _, indices = torch.sort(torch.abs(weight).view(-1))
        ne0_indices = indices[weight_mask.view(-1)[indices] != 0]
        length = int(ne0_indices.size(0) * percentage / 100)
        indices = ne0_indices[:length]
        weight_mask.view(-1)[indices] = 0

class ThreshTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.threshold(F.tanh(x), 0, 0)

class WeightProvider(object):
    def __init__(self, weights):
        self.weights = weights

    def update(self):
        pass

    def __call__(self):
        return self.weights

class GradientTracker(WeightProvider):
    def __init__(self, weights, alpha=0.01):
        super().__init__(weights)
        self.grads = None
        self.alpha = alpha

    def update(self):
        if self.grads is None:
            self.grads = [w.grad.clone() for w in self.weights]
        self.grads = [self.alpha * w.grad + (1 - self.alpha) * g for w, g in zip(self.weights, self.grads)]

    def __call__(self):
        return self.grads

def count_prunable_params(model):
    n_params = 0
    for m in model.modules():
        if isinstance(m, PruneLayer):
            for w in m.weight_masks():
                n_params += w.view(-1).size(0).data.cpu()[0]
    return n_params

def count_unpruned_params(model):
    n_params = 0
    for m in model.modules():
        if isinstance(m, PruneLayer):
            for w in m.weight_masks():
                n_params += w.sum().data.cpu()[0]
    return n_params

def count_params(model, type="prunable"):
    if type == "prunable":
        return count_prunable_params(model)
    elif type == "unpruned":
        return count_unpruned_params(model)

def find_activation(name):
    if name == "thresh_tanh":
        return ThreshTanh()
    else:
        return None

def prune_all(model, **kwargs):
    for l in model.modules():
        if isinstance(l, PruneLayer):
            l.prune(**kwargs)

class PruneLayer(SerializableModule):
    def __init__(self, config):
        super().__init__()
        self.prune_call = _methods[config.prune_method]
        self.config = config

    def _init_masks(self):
        if not self.config.use_cpu:
            self._weight_masks = [nn.Parameter(torch.ones(*w.size()).cuda(), requires_grad=False) for w in self.weights()]
        else:
            self._weight_masks = [nn.Parameter(torch.ones(*w.size()), requires_grad=False) for w in self.weights()]
        for i, w in enumerate(self._weight_masks):
            self.register_parameter("_mask{}".format(i), w)
        self.activation = find_activation(self.config.prune_activation)

    def weights(self):
        raise NotImplementedError

    def weight_masks(self):
        if self.activation is None:
            return self._weight_masks
        return [self.activation(w) for w in self._weight_masks]

    def prune(self, **kwargs):
        provider = kwargs.get("provider", WeightProvider(self.weights()))
        self.prune_call(provider(), self.weight_masks(), **kwargs)

    def forward(self, x):
        raise NotImplementedError

class PruneConv2d(PruneLayer):
    def __init__(self, conv_args, config, **kwargs):
        super().__init__(config)
        in_channels, out_channels, kernel_size = conv_args
        self.conv_kwargs = kwargs
        dummy_conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        if not config.use_cpu:
            dummy_conv = dummy_conv.cuda()
        self.weight = dummy_conv.weight
        self.bias = dummy_conv.bias
        self.kernel_size = dummy_conv.kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        if not kwargs.get("bias", True):
            kwargs["bias"] = None
        if "bias" in kwargs and kwargs["bias"]:
            del kwargs["bias"]
        self._init_masks()

    def weights(self):
        weights = [self.weight]
        if self.bias is not None:
            weights.append(self.bias)
        return weights

    def forward(self, x):
        weight = self.weight * self.weight_masks()[0]
        if self.bias is not None:
            bias = self.bias * self.weight_masks()[1]
            return F.conv2d(x, weight, bias, **self.conv_kwargs)
        else:
            return F.conv2d(x, weight, **self.conv_kwargs)

class PruneLinear(PruneLayer):
    def __init__(self, lin_args, config):
        super().__init__(config)
        in_features, out_features = lin_args
        dummy_linear = nn.Linear(in_features, out_features)
        if not config.use_cpu:
            dummy_linear = dummy_linear.cuda()
        self.weight = dummy_linear.weight
        self.bias = dummy_linear.bias
        self._init_masks()

    def weights(self):
        return [self.weight, self.bias]

    def forward(self, x):
        weight = self.weight * self.weight_masks()[0]
        bias = self.bias * self.weight_masks()[1]
        return F.linear(x, weight, bias)

_methods = dict(magnitude=prune_magnitude)