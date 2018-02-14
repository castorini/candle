from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

def prune_magnitude(weights, weight_mask, percentage=50):
    _, indices = torch.sort(weights.view(-1))
    length = int(indices.size(0) * percentage / 100)
    indices = indices[length:]
    weight_mask.view(-1)[indices] = 0
    weight_mask = 1 - weight_mask
    indices = indices[:length]
    weights.view(-1)[indices] = 0

class WeightProvider(object):
    def __init__(self, weights)
        self.weights = weights

    def update(self):
        pass

    def __call__(self):
        return self.weights

class GradientTracker(WeightProvider):
    def __init__(self, weights, alpha):
        super().__init__(weights)
        self.grads = None
        self.alpha = alpha

    def update(self):
        if self.grads is None:
            self.grads = [w.grad.clone() for w in self.weights]
        self.grads = [self.alpha * w.grad + (1 - self.alpha) * g for w, g in zip(self.weights, self.grads)]

    def __call__(self):
        return self.grads

class PruneLayer(SerializableModule):
    def __init__(self, child, config):
        super().__init__()
        self.prune_call = _methods[config.method]
        self.child = child
        self.weight_masks = [Variable(torch.ones(*w.size())) for w in self.weights]
        if not config.use_cpu:
            self.weight_masks = [w.cuda() for w in self.weight_masks]

    @property
    def weights(self):
        raise NotImplementedError

    def apply_mask(self):
        raise NotImplementedError

    def prune(self, **kwargs):
        provider = kwargs.get("provider", WeightProvider(self.weights))
        self.prune_call(provider(), self.weight_mask, **kwargs)

    def forward(self, x):
        self.apply_mask()
        return self.child(x)

class PruneConv2dLayer(PruneLayer):
    def __init__(self, child, config):
        super().__init__(child, config)

    @property
    def weights(self):
        return [self.child.weight, self.child.bias]

    def apply_mask(self):
        self.child.weight = self.child.weight * self.weight_masks[0]
        self.child.bias = self.child.bias * self.weight_masks[1]

_methods = dict(magnitude=prune_magnitude)