from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nested import Package

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

class Proxy(object):
    def __init__(self, layer):
        self.child = None
        self._proxy_layer = None
        self.layer = layer

    def parameters(self):
        return []

    @property
    def proxy_layer(self):
        return self._proxy_layer

    @proxy_layer.setter
    def proxy_layer(self, layer):
        self._proxy_layer = layer
        return self

    @property
    def param_options(self):
        if self.proxy_layer:
            return self.proxy_layer.param_options
        return {}

    def buffers(self):
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
    def __init__(self, layer, child):
        super().__init__(layer)
        self.child = child
        self.layer = layer

    @property
    def root(self):
        return self.child.root

    @property
    def proxy_layer(self):
        return self._proxy_layer

    @proxy_layer.setter
    def proxy_layer(self, layer):
        if self.child is not None:
            self.child.proxy_layer = layer
        self._proxy_layer = layer
        return self

    def print_info(self):
        if self.child is not None:
            self.child.print_info()

    def call(self, package, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.child is not None:
            package = self.child(*args, **kwargs)
            return self.call(package, **kwargs)
        else:
            return self.call(*args, **kwargs)

class FakeProxy(Proxy):
    def __init__(self, layer, parameters):
        super().__init__(layer)
        self.params = list(parameters)
        self.buffers = layer._buffers

    def parameters(self):
        return self.params

    def buffers(self):
        return self.buffers

    @property
    def sizes(self):
        return Package([p.size() for p in self.params])

    def __call__(self):
        raise ValueError("FakeProxy not callable!")

class IdentityProxy(Proxy):
    def __init__(self, layer, parameters):
        super().__init__(layer)
        self.package = Package(list(parameters))
        self._flattened_params = self.package.reify(flat=True)

    def parameters(self):
        return self._flattened_params

    @property
    def sizes(self):
        return self.package.size()

    def __call__(self):
        return self.package

class ProxyLayer(nn.Module):
    def __init__(self, weight_provider, registry=None):
        super().__init__()
        self.weight_provider = weight_provider
        self.weight_provider.proxy_layer = self
        self.output_proxy = None
        self.input_proxy = None
        self.registry = registry

        self._param_idx = 0
        self._register_all_params("weight_provider", weight_provider)

    def init_weights(self, init_fn):
        for param in self.weight_provider.root.parameters():
            try:
                if param.dim() > 1:
                    init_fn(param.data)
            except ValueError:
                pass

    def _register_all_params(self, proxy_type, proxy):
        self.registry.register_proxy(proxy_type, proxy)
        i = 0
        for i, parameter in enumerate(proxy.parameters()):
            self.register_parameter("proxy.{}".format(self._param_idx + i), parameter)
        self._param_idx += i + 1
        for name, buf in proxy.buffers():
            print(name)
            self.register_buffer(name, buf)

    def _find_provider(self, provider_type, provider):
        if isinstance(provider, provider_type):
            return provider
        if isinstance(provider, Proxy):
            return None
        return self._find_provider(provider_type, provider.child)

    def disable_hooks(self):
        self.weight_provider = self.weight_provider.root
        self.output_proxy = None
        self.input_proxy = None

    def find_provider(self, provider_type):
        return self._find_provider(provider_type, self.weight_provider)

    def hook_weight(self, weight_proxy, **kwargs):
        self.weight_provider = weight_proxy(self, self.weight_provider, **kwargs)
        self._register_all_params("weight_hook", self.weight_provider)
        return self.weight_provider

    def hook_output(self, output_proxy, **kwargs):
        self.output_proxy = output_proxy(self, self.output_proxy, **kwargs)
        self._register_all_params("output_hook", self.output_proxy)
        return self.output_proxy

    def hook_input(self, input_proxy, **kwargs):
        self.input_proxy = input_proxy(self, self.input_proxy, **kwargs)
        self._register_all_params("input_hook", self.input_proxy)
        return self.input_proxy

    def apply_input_hook(self, args):
        if self.input_proxy is None:
            return args
        return self.input_proxy(Package([list(args)])).reify()[0]

    def apply_output_hook(self, out):
        if self.output_proxy is None:
            return out
        return self.output_proxy(Package([out])).reify()[0]

    def forward(self, *args, **kwargs):
        args = self.apply_input_hook(args)
        out = self.on_forward(*args, **kwargs)
        out = self.apply_output_hook(out)
        return out

    @property
    def param_options(self):
        return dict(lr_scale=self.lr_scale)

    @property
    def lr_scale(self):
        return 1

    def on_forward(self, *args, **kwargs):
        raise NotImplementedError

class _ProxyConvNd(ProxyLayer):
    def __init__(self, weight_provider, conv_fn, stride=1, padding=0, dilation=1, **kwargs):
        super().__init__(weight_provider, **kwargs)
        sizes = weight_provider.sizes.reify()
        self._sizes = sizes
        self.bias = len(sizes) == 2
        self.kernel_size = sizes[0][2:]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_fn = conv_fn
        self._conv_kwargs = dict(dilation=dilation, padding=padding, stride=stride)
        if not self.bias:
            self._conv_kwargs["bias"] = None

    @property
    def lr_scale(self):
        # _sizes is [(C_out, C_in, kernel_size...), [bias]]
        n_inputs = np.prod(self.kernel_size) * self._sizes[0][1]
        n_units = np.prod(self.kernel_size) * self._sizes[0][0]
        scale = 1 / np.sqrt(1.5 / (n_inputs + n_units))
        return scale

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
        self._sizes = weight_provider.sizes.reify()

    @property
    def lr_scale(self):
        scale = 1 / np.sqrt(1.5 / (np.sum(self._sizes[0])))
        return scale

    @property
    def weight(self):
        return self.weight_provider().reify()[0]

    @property
    def bias(self):
        return self.weight_provider().reify()[1]

    def tie_weight(self, weight):
        root = self.weight_provider.root
        root._flattened_params[0] = weight
        root.layer.weight = weight
        root.package = Package(root._flattened_params)

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
        try:
            val = self.child(x, *args, **kwargs)
        finally:
            self.child._uninject()
        return val
