from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nested import Package

class Proxy(object):
    def __init__(self):
        self.child = None

    def parameters(self):
        return []

    @property
    def root(self):
        return self

    def print_info(self):
        pass

    @property
    def sizes(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class ProxyDecorator(Proxy):
    def __init__(self, child):
        super().__init__()
        self.child = child

    @property
    def root(self):
        return self.child.root

    def print_info(self):
        self.child.print_info()

    def call(self, package, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.child is not None:
            package = self.child(*args, **kwargs)
        return self.call(package, **kwargs)

class FakeProxy(Proxy):
    def __init__(self, parameters):
        super().__init__()
        self.params = list(parameters)

    def parameters(self):
        return self.params

    @property
    def sizes(self):
        return [p.size() for p in self.params]

    def __call__(self):
        raise ValueError("FakeProxy not callable!")

class IdentityProxy(Proxy):
    def __init__(self, parameters):
        super().__init__()
        self.package = Package(list(parameters))
        self._flattened_params = self.package.reify(flat=True)

    def parameters(self):
        return self._flattened_params

    @property
    def sizes(self):
        return self.package.size().reify()

    def __call__(self):
        return self.package

class ProxyLayer(nn.Module):
    def __init__(self, weight_provider, registry=None):
        super().__init__()
        self.weight_provider = weight_provider
        self.output_proxy = None
        self.input_proxy = None
        self.registry = registry

        self._param_idx = 0
        self._register_all_params("weight_provider", weight_provider)

    def _register_all_params(self, proxy_type, proxy):
        self.registry.register_proxy(proxy_type, proxy)
        for i, parameter in enumerate(proxy.parameters()):
            self.register_parameter("proxy.{}".format(self._param_idx + i), parameter)
        self._param_idx += i + 1

    def _find_provider(self, provider_type, provider):
        if isinstance(provider, provider_type):
            return provider
        if isinstance(provider, Proxy):
            return None
        return self._find_provider(provider_type, provider.child)

    def find_provider(self, provider_type):
        return self._find_provider(provider_type, self.weight_provider)

    def hook_weight(self, weight_proxy, **kwargs):
        self.weight_provider = weight_proxy(self.weight_provider, **kwargs)
        self._register_all_params("weight_hook", self.weight_provider)

    def hook_output(self, output_proxy, **kwargs):
        self.output_proxy = output_proxy(self.output_proxy, **kwargs)
        self._register_all_params("output_hook", self.output_proxy)

    def hook_input(self, input_proxy, **kwargs):
        self.input_proxy = input_proxy(self.input_proxy, **kwargs)
        self._register_all_params("input_hook", self.input_proxy)

    def apply_input_hook(self, *args):
        if self.input_proxy is None:
            return args
        return self.input_proxy(*args)

    def apply_output_hook(self, out):
        if self.output_proxy is None:
            return out
        return self.output_proxy(out)

    def forward(self, *args, **kwargs):
        if self.input_proxy is not None:
            args = self.input_proxy(*args)
        out = self.on_forward(*args, **kwargs)
        if self.output_proxy is not None:
            out = self.output_proxy(out)
        return out

    def on_forward(self, *args, **kwargs):
        raise NotImplementedError

class _ProxyConvNd(ProxyLayer):
    def __init__(self, weight_provider, conv_fn, stride=1, padding=0, dilation=1, **kwargs):
        super().__init__(weight_provider, **kwargs)
        sizes = weight_provider.sizes
        self.bias = len(sizes) == 2
        self.kernel_size = sizes[0][2:]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_fn = conv_fn
        self._conv_kwargs = dict(dilation=dilation, padding=padding, stride=stride)
        if not self.bias:
            self._conv_kwargs["bias"] = None

    def on_forward(self, x):
        weights = self.weight_provider().reify()
        return self.conv_fn(x, *weights, **self._conv_kwargs)

class ProxyConv3d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv3d, **kwargs)

class ProxyConv2d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv2d, **kwargs)

class ProxyConv1d(_ProxyConvNd):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, F.conv1d, **kwargs)

class ProxyLinear(ProxyLayer):
    def __init__(self, weight_provider, **kwargs):
        super().__init__(weight_provider, **kwargs)

    def on_forward(self, x):
        weights = self.weight_provider().reify()
        return F.linear(x, *weights)

class ProxyRNNBase(nn.modules.rnn.RNNBase):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super().__init__(mode, input_size, hidden_size, num_layers, bias, batch_first,
            dropout, bidirectional)
        self.weights = None

    def _inject(self, weights):
        self.weights = weights

    def _uninject(self):
        self.weights = None

    @property
    def all_weights(self):
        if not self.weights:
            return super().all_weights
        return self.weights

class ProxyRNN(ProxyLayer):
    def __init__(self, child, weight_provider, **kwargs):
        super().__init__(weight_provider, **kwargs)
        self.child = child
        self.child.flatten_parameters = self._null_fn # flatten_parameters hack

    def _null_fn(*args, **kwargs):
        return

    def on_forward(self, x, *args, **kwargs):
        self.child._inject(self.weight_provider().reify())
        val = self.child(x, *args, **kwargs)
        self.child._uninject()
        return val
