from torch.autograd import Variable
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from candle.context import Context
from candle.estimator import *
from candle.nested import *
from candle.proxy import *

def prune_global_norm(model, percentage, p=1):
    global_scores = []
    for module in filter(lambda x: isinstance(x, nn.Conv2d), model.modules()):
        module.pruned_indices = []
        norms = module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1)
        norm_sum = torch.sum(norms)
        scores = norms / norm_sum
        local_scores = []
        for idx in range(module.weight.size(0)):
            local_scores.append((scores[idx].item(), idx, module))
        local_scores.sort(key=lambda x: x[0])
        global_scores.extend(local_scores[:-1])
    global_scores.sort(key=lambda x: x[0])
    for _, idx, module in global_scores[:int(percentage * len(global_scores))]:
        module.weight.data[idx, ...] = 0
        module.bias.data[idx] = 0
        module.pruned_indices.append(idx)

def prune_local_norm(model, percentage, p=1):
    for module in filter(lambda x: isinstance(x, nn.Conv2d), model.modules()):
        module.pruned_indices = []
        norms = module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1)
        norm_sum = torch.sum(norms)
        scores = norms / norm_sum
        local_scores = []
        for idx in range(module.weight.size(0)):
            local_scores.append((scores[idx].item(), idx, module))
        local_scores.sort(key=lambda x: x[0])
        for _, idx, module in local_scores[:int(percentage * len(local_scores[:-1]))]:
            module.weight.data[idx, ...] = 0
            module.bias.data[idx] = 0
            module.pruned_indices.append(idx)

def compute_multiply_factor(model):
    modules = filter(lambda x: isinstance(x, nn.Conv2d), model.modules())
    mults = 0
    for module in modules:
        factor_in = module.weight.size(1)
        factor_out = module.weight.size(0)
        if hasattr(module, "pruned_indices"):
            factor_out -= len(module.pruned_indices)
        mults += factor_in * factor_out
    return mults
