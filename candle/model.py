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
        self.config = prune_cfg
        self.kwargs = kwargs

    def _make_prune_module(self, module, **kwargs):
        kwargs["prunable"] = kwargs.get("prunable", False)
        if isinstance(module, candle.PruneLayer):
            return module
        if isinstance(module, nn.modules.conv._ConvNd):
            in_c, out_c, k_size = module.in_channels, module.out_channels, module.kernel_size
            if isinstance(module, nn.Conv2d):
                return candle.PruneConv2d((in_c, out_c, k_size), self.config, **kwargs)
            elif isinstance(module, nn.Conv1d):
                return candle.PruneConv1d((in_c, out_c, k_size), self.config, **kwargs)
            elif isinstance(module, nn.Conv3d):
                return candle.PruneConv3d((in_c, out_c, k_size), self.config, **kwargs)
        elif isinstance(module, nn.Linear):
            in_feats, out_feats = module.in_features, module.out_features
            return candle.PruneLinear((in_feats, out_feats), self.config, **kwargs)
        elif isinstance(module, nn.modules.rnn.RNNBase):
            mode, input_size, hidden_size = module.mode, module.input_size, module.hidden_size
            num_layers, bias, batch_first = module.num_layers, module.bias, module.batch_first
            bidirectional, dropout = module.bidirectional, module.dropout
            base = candle.PruneRNNBase(mode, input_size, hidden_size, num_layers=num_layers, 
                bias=bias, batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)
            return candle.PruneRNN(base, self.config, **kwargs)

    def pruned(self, module, **kwargs):
        kwargs["prunable"] = True
        kwargs_cpy = self.kwargs.copy()
        kwargs_cpy.update(kwargs)
        return self._make_prune_module(module, **kwargs_cpy)

    def binarized(self, module, binary_activations=True, **kwargs):
        kwargs_cpy = self.kwargs.copy()
        kwargs_cpy.update(kwargs)
        module = self._make_prune_module(module, **kwargs_cpy)
        module.register_parameter_hook(candle.binarize)
        if binary_activations:
            module.register_mutable_forward_hook(candle.binarize)
        return module

# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/gradients_impl.py#L924
def hessian_vector_product(y, x, v):
    v = v.detach()
    grads = ag.grad(y, x, create_graph=True)[0]
    elem_prods = torch.sum(v * grads)
    return ag.grad(elem_prods, x, create_graph=True)[0], grads