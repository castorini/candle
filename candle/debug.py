import enum

import torch
import torch.autograd as ag

from .nested import *
from .proxy import *

def debug_print(name=""):
    def arg_wrap(function):
        def print_wrap(*args, **kwargs):
            x = function(*args, **kwargs)
            print(f"{name}", x.reify() if isinstance(x, Package) else x)
            return x
        return print_wrap
    return arg_wrap

class DebugType(enum.Enum):
    FORWARD_IN       = 1 << 0
    FORWARD_OUT      = 1 << 1
    BACKWARD_OUT     = 1 << 2
    WEIGHTS          = 1 << 3
    BACKWARD_WEIGHTS = 1 << 4

class DebugHook(ProxyDecorator):
    def __init__(self, layer, child, name=""):
        super().__init__(layer, child)
        self.name = name

    @property
    def sizes(self):
        if self.child:
            return self.child.sizes
        return None

    def debug(self, package):
        raise NotImplementedError

    @staticmethod
    @property
    def type():
        raise NotImplementedError

    def call(self, package, **kwargs):
        self.debug(package)
        return package

class PrintHook(DebugHook):
    def __init__(self, layer, child, name=""):
        super().__init__(layer, child, name)

    def debug(self, package):
        print(f"{self.name} {package.cpu().detach().numpy().reify()}")

class MeanMinMaxHook(DebugHook):
    def __init__(self, layer, child, name=""):
        super().__init__(layer, child, name)

    def debug(self, package):
        mean = torch.cat(package.view(-1).reify(flat=True)).mean().data[0]
        min_ = min(package.min().data[0].reify(flat=True))
        max_ = max(package.max().data[0].reify(flat=True))
        print(f"{self.name} mean: {mean:<6.5} min: {min_:<6.5} max: {max_:<6.5}")

class _DebugFunction(ag.Function):
    @staticmethod
    def forward(ctx, x, hook):
        ctx.hook = hook
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.hook.debug(Package([grad_output]))
        return grad_output, None

debug_fn = _DebugFunction.apply

class BackwardWeightHook(ProxyDecorator):
    def __init__(self, layer, child, hook=None):
        super().__init__(layer, child)
        self.hook = hook

    @property
    def sizes(self):
        return self.child.sizes

    def call(self, package, **kwargs):
        return package.apply_fn(lambda x: debug_fn(x, self.hook))

class BackwardOutHookAdapter(object):
    def __init__(self, layer, hook_type, **kwargs):
        self.hook = hook_type(layer, None, **kwargs)
    
    def __call__(self, ctx, grad_input, grad_output):
        self.hook.debug(Package(list(grad_output)))

def debug_layer(layer, hook_types, type=0, **kwargs):
    for hook_type in hook_types:
        if type & DebugType.FORWARD_IN.value:
            layer.hook_input(hook_type, **kwargs)
        if type & DebugType.FORWARD_OUT.value:
            layer.hook_output(hook_type, **kwargs)
        if type & DebugType.WEIGHTS.value:
            layer.hook_weight(hook_type, **kwargs)
        if type & DebugType.BACKWARD_OUT.value:
            layer.register_backward_hook(BackwardOutHookAdapter(layer, hook_type, **kwargs))
        if type & DebugType.BACKWARD_WEIGHTS.value:
            hook = hook_type(layer, None, **kwargs)
            layer.hook_weight(BackwardWeightHook, hook=hook)

