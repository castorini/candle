import torch.autograd as ag
import torch
import torch.nn as nn

import candle

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

class Context(object):
    def __init__(self, prune_cfg, **kwargs):
        self.kwargs = dict(prune_method=prune_cfg.prune_method, prune_trainable=prune_cfg.prune_trainable,
            prune_activation=prune_cfg.prune_activation)
        self.kwargs.update(kwargs)

    def _make_prune_module(self, module, **kwargs):
        kwargs["prunable"] = kwargs.get("prunable", False)
        if isinstance(module, candle.PruneLayer):
            return module
        if isinstance(module, nn.modules.conv._ConvNd):
            in_c, out_c, k_size = module.in_channels, module.out_channels, module.kernel_size
            stride, padding = module.stride, module.padding
            kwargs["stride"] = stride
            kwargs["padding"] = padding
            if isinstance(module, nn.Conv2d):
                return candle.PruneConv2d(module, **kwargs)
            elif isinstance(module, nn.Conv1d):
                return candle.PruneConv1d(module, **kwargs)
            elif isinstance(module, nn.Conv3d):
                return candle.PruneConv3d(module, **kwargs)
        elif isinstance(module, nn.Linear):
            return candle.PruneLinear(module, **kwargs)
        elif isinstance(module, nn.modules.rnn.RNNBase):
            mode, input_size, hidden_size = module.mode, module.input_size, module.hidden_size
            num_layers, bias, batch_first = module.num_layers, module.bias, module.batch_first
            bidirectional, dropout = module.bidirectional, module.dropout
            base = candle.PruneRNNBase(mode, input_size, hidden_size, num_layers=num_layers, 
                bias=bias, batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)
            return candle.PruneRNN(base, **kwargs)

    def pruned(self, module, **kwargs):
        kwargs["prunable"] = True
        kwargs_cpy = self.kwargs.copy()
        kwargs_cpy.update(kwargs)
        return self._make_prune_module(module, **kwargs_cpy)

    def binarized(self, module, **kwargs):
        kwargs_cpy = self.kwargs.copy()
        kwargs_cpy.update(kwargs)
        module = self._make_prune_module(module, **kwargs_cpy)
        module.register_parameter_hook(candle.binarize)
        module.binary = True
        for weight in module.weights():
            weight.data.uniform_(-1, 1)
        return module
