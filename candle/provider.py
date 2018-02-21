import torch
import torch.autograd as ag

from .model import hessian_vector_product as hvp

class WeightProvider(object):
    def __init__(self, layer, average=True):
        self.weights = layer.weights()
        self.average = average
        self._reset()

    def _reset(self):
        if self.average:
            self.sum_weights = [0] * len(self.weights)

    def update(self):
        if self.average:
            self.sum_weights = [sw + w for w, sw in zip(self.weights, self.sum_weights)]

    def __call__(self, loss):
        if self.average:
            weights = self.sum_weights
            self._reset()
            return weights
        else:
            return self.weights

class RandomWeightProvider(WeightProvider):
    def __init__(self, layer):
        super().__init__(layer)

    def __call__(self, loss):
        weights = [w.clone() for w in self.weights]
        return [w.uniform_() for w in weights]

class WeightMaskGradientProvider(WeightProvider):
    def __init__(self, layer, negative=True):
        super().__init__(layer)
        self.weight_masks = layer.weight_masks()
        self.negative = negative
        self._reset()

    def _reset(self):
        self.grads = [0] * len(self.weights)
        self.count = 0

    def update(self):
        self.grads = [g + w.grad for w, g in zip(self.weights, self.grads)]
        self.count += 1

    def __call__(self, loss):
        sign = -1 if self.negative else 1
        grads = [sign * g / self.count for g in self.grads]
        self._reset()
        return grads

class InverseGradientTracker(WeightProvider):
    def __init__(self, layer, alpha=0.01):
        super().__init__(layer)
        self.grads = None
        self.alpha = alpha

    def update(self):
        if self.grads is None:
            self.grads = [w.grad.clone() for w in self.weights]
        self.grads = [self.alpha * w.grad + (1 - self.alpha) * g for w, g in zip(self.weights, self.grads)]

    def __call__(self, loss):
        return [1 / (torch.abs(g) + 1E-6) for g in self.grads]

class OptimalBrainDamageApproximator(WeightProvider):
    def __init__(self, layer):
        super().__init__(layer, average=False)

    def _recursive_sum(self, x):
        s = 0
        for elem in x:
            if isinstance(elem, list) or isinstance(elem, tuple):
                s = s + self._recursive_sum(elem)
            else:
                s = s + torch.sum(elem)
        return s

    def __call__(self, loss):
        if loss is None:
            raise ValueError("Loss is required for this weight provider!")
        grads = ag.grad(loss, self.weights, create_graph=True)
        grad_sum = self._recursive_sum(grads)
        hessians = ag.grad(grad_sum, self.weights, create_graph=True)
        return [h * w**2 for w, h in zip(self.weights, hessians)]
