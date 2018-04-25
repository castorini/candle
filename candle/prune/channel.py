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

class ChannelDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.Tensor(x.size(1)).fill_(1 - self.drop_prob).bernoulli_()
        if x.is_cuda:
            mask = mask.cuda()
        mask = mask.unsqueeze(0)
        for _ in x.size()[2:]:
            mask = mask.unsqueeze(-1)
        x = (x * mask.expand_as(x)) / (1 - self.drop_prob)
        return x

def prune_qrnn(linear, percentage, p=1):
    linear.pruned_indices = []
    Z_w, F_w, O_w = linear.module.weight_raw.chunk(3, dim=0)
    Z_b, F_b, O_b = linear.module.bias.chunk(3, dim=0)
    Z_n, F_n, O_n = linear.module.weight_raw.norm(p=p, dim=1).chunk(3, dim=0)
    Z_max, F_max, O_max = Z_n.max(), F_n.max(), O_n.max()
    Z_n, F_n, O_n = Z_n / Z_max, F_n / F_max, O_n / O_max
    scores = Z_n + F_n + O_n
    local_scores = []
    for idx in range(scores.size(0)):
        local_scores.append((scores[idx].item(), idx))
    
    indices = []
    local_scores.sort(key=lambda x: x[0])
    for _, idx in local_scores[:int(percentage * len(local_scores[:-1]))]:
        indices.append(idx)
        idx1 = idx
        idx2 = idx + scores.size(0)
        idx3 = idx + scores.size(0) * 2
        linear.module.weight_raw.data[idx1, ...] = 0
        linear.module.bias.data[idx1] = 0
        linear.module.weight_raw.data[idx2, ...] = 0
        linear.module.bias.data[idx2] = 0
        linear.module.weight_raw.data[idx3, ...] = 0
        linear.module.bias.data[idx3] = 0
        linear.pruned_indices.extend([idx1, idx2, idx3])
    indices.sort()
    print(indices)

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
    modules = filter(lambda x: isinstance(x, nn.Conv2d) or isinstance(x, nn.MaxPool2d), model.modules())
    mults = 0
    scale = 1
    last_pruned = 0
    for module in modules:
        if isinstance(module, nn.MaxPool2d):
            scale /= module.kernel_size**2
            continue
        factor_in = module.weight.size(1)
        factor_out = module.weight.size(0)
        factor_in -= last_pruned
        if hasattr(module, "pruned_indices"):
            last_pruned = len(module.pruned_indices)
            factor_out -= last_pruned
        mults += scale * factor_in * factor_out
    return mults
