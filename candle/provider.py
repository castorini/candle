import torch
import torch.autograd as ag

class WeightProvider(object):
    def __init__(self, layer, average=False):
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

class LossApproximator(WeightProvider):
    def __init__(self, layer, alpha=0.01):
        super().__init__(layer, average=False)

    def __call__(self):
        grads = [w.grad for w in self.weights]
        return [g * w for w, g in zip(self.weights, grads)]
