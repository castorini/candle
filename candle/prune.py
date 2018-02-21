from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import SerializableModule
from .provider import WeightProvider

def prune_magnitude(weights, weight_masks, **kwargs):
    percentage = kwargs.get("percentage", 50)
    min_length = kwargs.get("min_length", 0)
    for weight, weight_mask in zip(weights, weight_masks):
        _, indices = torch.sort(torch.abs(weight).view(-1))
        ne0_indices = indices[weight_mask.view(-1)[indices] != 0]
        if ne0_indices.size(0) < min_length:
            continue
        length = int(ne0_indices.size(0) * percentage / 100)
        indices = ne0_indices[:length]
        if indices.size(0) > 0:
            weight_mask.data.view(-1)[indices.data] = 0

class CliffFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_variables
        grad_output[grad_output > 0] *= alpha
        return grad_output * x, None

cliff = CliffFunction.apply

class ReLUCliff(SerializableModule):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = Variable(torch.Tensor([alpha]).cuda(), requires_grad=False)

    def forward(self, x):
        return cliff(x, self.alpha)

def list_params(model, train_prune=False):
    params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, WeightMaskParameter) and train_prune:
            params.append(param)
        elif not isinstance(param, WeightMaskParameter):
            params.append(param)
    return params

def count_prunable_params(model):
    n_params = 0
    for m in list_prune_layers(model):
        for w in m.weight_masks():
            n_params += w.view(-1).size(0).data.cpu()[0]
    return n_params

def count_unpruned_params(model):
    n_params = 0
    for m in list_prune_layers(model):
        for w in m.weight_masks():
            n_params += w.sum().data.cpu()[0]
    return n_params

def count_params(model, type="prunable"):
    if type == "prunable":
        return count_prunable_params(model)
    elif type == "unpruned":
        return count_unpruned_params(model)

def list_prune_layers(model):
    for m in model.modules():
        if isinstance(m, PruneLayer):
            yield m

def normalize_masks(model, soft=False):
    for m in list_prune_layers(model):
        for w in m.weight_masks():
            if soft:
                w.data.clamp_(0, 1)
            else:
                w.data.clamp_(0, 1).round_()

def find_activation(name):
    if name == "cliff":
        return ReLUCliff()
    else:
        return None

def prune_all(model, **kwargs):
    for l in list_prune_layers(model):
        l.prune(**kwargs)

def apply_all_hooks(model):
    for l in list_prune_layers(model):
        l.apply_hook()

def update_all(model, **kwargs):
    for l in list_prune_layers(model):
        l.provider.update(**kwargs)

def remove_activations(model):
    for l in list_prune_layers(model):
        l.activation = None

def restore_activations(model):
    for l in list_prune_layers(model):
        l.activation = find_activation(l.config.prune_activation)

class WeightMaskParameter(nn.Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PruneLayer(SerializableModule):
    def __init__(self, config, provider=None, min_length=0, prunable=True):
        super().__init__()
        self.prune_call = _methods[config.prune_method]
        self.config = config
        self.provider = provider
        self.hook = None
        self.mutable_forward_hook = None
        self.min_length = min_length
        self.prunable = prunable

    def _init(self):
        self._init_masks()
        self._init_provider()

    def _init_masks(self):
        if not self.config.use_cpu:
            self._weight_masks = [WeightMaskParameter(torch.ones(*w.size()).cuda(),
                requires_grad=self.config.prune_trainable) for w in self.weights()]
        else:
            self._weight_masks = [WeightMaskParameter(torch.ones(*w.size()),
                requires_grad=self.config.prune_trainable) for w in self.weights()]
        for i, w in enumerate(self._weight_masks):
            self.register_parameter("_mask{}".format(i), w)
        self.activation = find_activation(self.config.prune_activation)

    def _init_provider(self):
        if self.provider is None:
            self.provider = WeightProvider
        self.provider = self.provider(self)

    def register_parameter_hook(self, hook):
        self.hook = hook

    def register_mutable_forward_hook(self, hook):
        self.mutable_forward_hook = hook

    def weights(self):
        raise NotImplementedError

    def hook_weight(self, weight):
        if not self.hook:
            return weight
        return self.hook(weight)

    def apply_hook(self):
        for w in self.weights():
            w.data = self.hook_weight(w).data

    def hooked_weights(self):
        return [self.hook_weight(w) for w in weights]

    def weight_masks(self):
        if self.activation is None:
            return self._weight_masks
        return [self.activation(w) for w in self._weight_masks]

    def prune(self, loss=None, **kwargs):
        if not self.prunable:
            return
        self.prune_call(self.provider(loss), self.weight_masks(), min_length=self.min_length, **kwargs)

    def on_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        out = self.on_forward(*args, **kwargs)
        if self.mutable_forward_hook:
            out = self.mutable_forward_hook(out)
        return out

class PruneRNNBase(nn.modules.rnn.RNNBase):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super().__init__(mode, input_size, hidden_size, num_layers, bias, batch_first,
            dropout, bidirectional)
        self.hook_weight = None
        self.weight_masks = None

    def _inject(self, hook_weight, weight_masks):
        self.hook_weight = hook_weight
        self.weight_masks = weight_masks

    def _uninject(self):
        self.hook_weight = None
        self.weight_masks = None

    @property
    def all_weights(self):
        if not self.hook_weight:
            return super().all_weights
        all_weights = []
        for i, weights in enumerate(self._all_weights):
            weight_list = []
            for j, weight in enumerate(weights):
                idx = i * len(weights) + j
                w = self.hook_weight(getattr(self, weight)) * self.weight_masks[idx]
                weight_list.append(w)
            all_weights.append(weight_list)
        return all_weights

    def weights(self):
        return list(self.parameters())

class PruneRNN(PruneLayer):
    def __init__(self, child, config, **kwargs):
        super().__init__(config, **kwargs)
        self.child = child
        self._init()

    def weights(self):
        return self.child.weights()

    def on_forward(self, x, *args, **kwargs):
        self.child._inject(self.hook_weight, self.weight_masks())
        val = self.child(x, *args, **kwargs)
        self.child._uninject()
        return val

class _PruneConvNd(PruneLayer):
    def __init__(self, conv_args, config, conv_cls, conv_fn, provider=None, 
            min_length=0, prunable=False, **kwargs):
        super().__init__(config, provider, min_length)
        in_channels, out_channels, kernel_size = conv_args
        self.conv_kwargs = kwargs
        dummy_conv = conv_cls(in_channels, out_channels, kernel_size, **kwargs)
        if not config.use_cpu:
            dummy_conv = dummy_conv.cuda()
        self.weight = dummy_conv.weight
        self.bias = dummy_conv.bias
        self.kernel_size = dummy_conv.kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        if not kwargs.get("bias", True):
            kwargs["bias"] = None
        if "bias" in kwargs and kwargs["bias"]:
            del kwargs["bias"]
        self.conv_fn = conv_fn
        self._init()

    def weights(self):
        weights = [self.weight]
        if self.bias is not None:
            weights.append(self.bias)
        return weights

    def on_forward(self, x):
        weight = self.hook_weight(self.weight) * self.weight_masks()[0]
        if self.bias is not None:
            bias = self.hook_weight(self.bias) * self.weight_masks()[1]
            return self.conv_fn(x, weight, bias, **self.conv_kwargs)
        else:
            return self.conv_fn(x, weight, **self.conv_kwargs)

class PruneConv3d(_PruneConvNd):
    def __init__(self, conv_args, config, provider=None, min_length=0, **kwargs):
        super().__init__(conv_args, config, nn.Conv3d, F.conv3d, provider=provider, min_length=min_length, **kwargs)

class PruneConv2d(_PruneConvNd):
    def __init__(self, conv_args, config, provider=None, min_length=0, **kwargs):
        super().__init__(conv_args, config, nn.Conv2d, F.conv2d, provider=provider, min_length=min_length, **kwargs)

class PruneConv1d(_PruneConvNd):
    def __init__(self, conv_args, config, provider=None, min_length=0, **kwargs):
        super().__init__(conv_args, config, nn.Conv1d, F.conv1d, provider=provider, min_length=min_length, **kwargs)

class PruneLinear(PruneLayer):
    def __init__(self, lin_args, config, **kwargs):
        super().__init__(config, **kwargs)
        in_features, out_features = lin_args
        dummy_linear = nn.Linear(in_features, out_features)
        if not config.use_cpu:
            dummy_linear = dummy_linear.cuda()
        self.weight = dummy_linear.weight
        self.bias = dummy_linear.bias
        self._init()

    def weights(self):
        return [self.weight, self.bias]

    def on_forward(self, x):
        weight = self.hook_weight(self.weight) * self.weight_masks()[0]
        bias = self.hook_weight(self.bias) * self.weight_masks()[1]
        return F.linear(x, weight, bias)

_methods = dict(magnitude=prune_magnitude)