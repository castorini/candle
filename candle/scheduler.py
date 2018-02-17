import numpy as np

class PruningScheduler(object):
    def __init__(self, begin_idx=0, end_idx=np.inf, end_y=0):
        self._idx = 0
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.end_y = end_y

    def step(self):
        self._idx += 1

    def apply(self, target):
        percentage = self.compute_rate()
        target.prune(percentage=percentage)
        self.step()

    def compute_rate(self):
        y = self._compute_rate()
        if y <= self.end_y:
            return 0
        elif self._idx >= self.end_idx:
            return 0
        elif self._idx < self.begin_idx:
            return 0
        return y

    @property
    def idx(self):
        return self._idx - self.begin_idx

    def _compute_rate(self):
        raise NotImplementedError

class SinglePruningScheduler(PruningScheduler):
    def __init__(self, set_value, **kwargs):
        super().__init__(**kwargs)
        self.set_value = set_value

    def _compute_rate(self):
        if self.idx == 0:
            val = self.set_value
            self.set_value = 0
            return val
        else:
            return 0

class ExponentialPruningScheduler(PruningScheduler):
    def __init__(self, init_value, const=2, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.const = const

    def _compute_rate(self):
        return self.const * self.init_value**self.idx

class LinearPruningScheduler(PruningScheduler):
    def __init__(self, init_value, slope=0.1, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.slope = slope

    def _compute_rate(self):
        return self.init_value + (-self.slope * self.idx)
