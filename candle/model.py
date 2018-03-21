from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .context import *
from .estimator import Function
from .nested import *
from .proxy import *

# TODO
class ShuffleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, type=nn.LSTM, hidden_factor=2, input_factor=2, **kwargs):
        super().__init__()
        if hidden_size % hidden_factor != 0 or input_size % input_factor != 0:
            raise ValueError("Input and hidden sizes must be divisible by reduction factors")
        self.rnns = []
        for i in range(input_factor):
            layer = [type(input_size // input_factor, hidden_size // hidden_factor, 1, **kwargs) for _ in num_layers]
            setattr(self, f"_rnns{i}", nn.ModuleList(layer))
            self.rnns.append(layer)

    def forward(self, x, **kwargs):
        outputs = []
        for rnns in zip(*self.rnns):
            output = []
            x_list = x.split(len(rnns), int(not rnn.batch_first))
            for rnn, input in zip(rnns, x_list):
                output.append(rnn(input, **kwargs))
            outputs.append(output)
        return outputs

