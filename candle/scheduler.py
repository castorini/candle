import numpy as np

class Scheduler(object):
    def __init__(self, begin_idx=0, end_idx=np.inf, end_y=0):
        self._idx = 0
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.end_y = end_y

    def step(self):
        self._idx += 1

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

class SingleScheduler(Scheduler):
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

class ExponentialScheduler(Scheduler):
    def __init__(self, init_value, const=2, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.const = const

    def _compute_rate(self):
        return self.const * self.init_value**self.idx

class LinearScheduler(Scheduler):
    def __init__(self, init_value, slope=-0.1, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.slope = slope

    def _compute_rate(self):
        return self.init_value + (self.slope * self.idx)

class LogisticFunction(object):
    def __init__(self, k, scale, shift, c, begin_idx=0):
        self.k = k
        self.scale = scale
        self.shift = shift
        self.c = c
        self.idx = begin_idx

    def __call__(self, x):
        return float(self.k / (1 + np.exp(-(self.scale * x - self.shift)) + 1E-8) + self.c)

    def next(self):
        val = self(self.idx)
        self.idx += 1
        return val

    @classmethod
    def interpolated(cls, begin_idx, end_idx, begin_val, end_val, eps=0.00247):
        assert end_idx >= begin_idx
        c = begin_val
        k = end_val - begin_val
        eps = eps * np.sign(end_val - begin_val)
        scale = (np.log(k / eps - 1) - np.log(k / (k - eps) - 1)) / (end_idx - begin_idx)
        shift = np.log(k / eps - 1) + scale * begin_idx
        return cls(k, scale, shift, c, begin_idx=begin_idx)