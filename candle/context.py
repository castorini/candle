import argparse
import itertools

import torch
import torch.nn as nn

from .debug import *
from .proxy import *

def read_cli_config():
    parser = argparse.ArgumentParser()
    return parser.parse_known_args()[0]

def find_modules(net, *types):
    modules = []
    for module in net.modules():
        for t in types:
            if isinstance(module, t):
                modules.append(module)
                break
    return modules

class ProxyRegistry(object):
    def __init__(self):
        self.table = {}
        self.proxies = []

    def register_proxy(self, proxy_type, proxy):
        if proxy_type not in self.table:
            self.table[proxy_type] = {}
        if type(proxy) not in  self.table[proxy_type]:
            self.table[proxy_type][type(proxy)] = []
        self.table[proxy_type][type(proxy)].append(proxy)
        self.proxies.append(proxy)

    def find_all(self, proxy_type=None, proxy_class=None):
        if proxy_type is None:
            return self.proxies
        proxies = self.table.get(proxy_type, {}).values()
        proxies = list(itertools.chain.from_iterable(proxies))
        if proxy_class is None:
            return proxies
        return list(filter(lambda x: isinstance(x, proxy_class), proxies))

class Memoizer(object):
    def __init__(self):
        self.cache = {}

    def wrap(self, function, *args, **kwargs):
        return lambda: function(*args, **kwargs)

    def delete(self, name):
        try:
            del self.cache[name]
        except:
            pass

    def __call__(self, key, not_present_fn, refresh=False):
        if refresh or key not in self.cache:
            self.cache[key] = not_present_fn()
        return self.cache[key]

class Context(object):
    def __init__(self, config=None, **kwargs):
        self._cfg_kwargs = vars(config) if config else {}
        self._cfg_kwargs.update(kwargs)
        self.registry = ProxyRegistry()
        self.layers = []
        self.torch_modules = []
        self.opt_params = []
        self.cache = Memoizer()

    def build_provider(self, layer):
        return IdentityProxy(layer, layer.parameters())

    def list_proxies(self, proxy_type=None, proxy_class=None):
        return self.cache("proxies.{}.{}".format(proxy_type, proxy_class),
            lambda: self.registry.find_all(proxy_type, proxy_class))

    def list_providers(self):
        return self.list_proxies("weight_provider")

    def add_parameter(self, parameter):
        self.opt_params.append(parameter)

    def print_info(self):
        for proxy in self.list_proxies():
            proxy.print_info()

    def list_params(self, filter_fn=None, include_opt=True):
        all_proxies = self.list_proxies()
        if filter_fn is None:
            lst = list(dict(params=p.parameters(), **p.param_options) for p in all_proxies)
        else:
            lst = list(dict(params=p.parameters(), **p.param_options) for p in filter(filter_fn, all_proxies))
        if include_opt:
            lst.extend(self.opt_params)
        return lst

    def list_buffers(self, filter_fn=None):
        all_proxies = self.list_proxies()
        if filter_fn is None:
            return list(itertools.chain.from_iterable(p.buffers() for p in all_proxies))
        return list(itertools.chain.from_iterable(p.buffers() for p in filter(filter_fn, all_proxies)))

    def compose(self, layer, **cfg):
        if isinstance(layer, ProxyLayer):
            return layer

        kwargs = dict(registry=self.registry)
        provider = self.build_provider(layer)
        self.torch_modules.append(layer)

        if isinstance(layer, nn.modules.conv._ConvNd):
            stride, padding, dilation = layer.stride, layer.padding, layer.dilation
            kwargs["stride"] = stride
            kwargs["padding"] = padding
            kwargs["dilation"] = dilation
            if isinstance(layer, nn.Conv3d):
                return ProxyConv3d(provider, **kwargs)
            elif isinstance(layer, nn.Conv2d):
                return ProxyConv2d(provider, **kwargs)
            elif isinstance(layer, nn.Conv1d):
                return ProxyConv1d(provider, **kwargs)
            else:
                raise ValueError("Unsupported!")
        elif isinstance(layer, nn.Linear):
            return ProxyLinear(provider, **kwargs)
        elif isinstance(layer, nn.modules.rnn.RNNBase):
            mode, input_size, hidden_size = layer.mode, layer.input_size, layer.hidden_size
            num_layers, bias, batch_first = layer.num_layers, layer.bias, layer.batch_first
            bidirectional, dropout = layer.bidirectional, layer.dropout
            base = ProxyRNNBase(mode, input_size, hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)
            provider = IdentityProxy(layer, layer.all_weights)
            return ProxyRNN(base, provider, **kwargs)
        else:
            raise ValueError("Unsupported!")

    def wrap(self, layer, **kwargs):
        cfg = self._cfg_kwargs.copy()
        cfg.update(kwargs)
        wrapped_layer = self.compose(layer, **cfg)
        self.layers.append(wrapped_layer) # TODO: insert per-layer hyperparams (mask decay, etc) here if needed
        return wrapped_layer

    def debug(self, layer, hook_types, type, **kwargs):
        debug_layer(layer, hook_types, type, **kwargs)
        return layer

    def bypass(self, layer):
        self.registry.register_proxy("fake", FakeProxy(layer, layer.parameters()))
        self.torch_modules.append(layer)
        return layer

    def disable_hooks(self):
        for layer in self.layers:
            layer.disable_hooks()

class MixedContext(object):
    def __init__(self, config, *contexts, **kwargs):
        pass # TODO
