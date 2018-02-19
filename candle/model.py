import torch
import torch.nn as nn

import candle

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class Context(object):
    def __init__(self, prune_cfg, **kwargs):
        self.config = prune_cfg
        self.kwargs = kwargs

    def _make_prune_module(self, module, **kwargs):
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

    def pruned(self, module, **kwargs):
        kwargs.update(self.kwargs)
        return self._make_prune_module(module, **kwargs)

    def binarized(self, module, **kwargs):
        kwargs.update(self.kwargs)
        module = self._make_prune_module(module, **kwargs)
        module.register_parameter_hook(candle.binarize)
        return module

