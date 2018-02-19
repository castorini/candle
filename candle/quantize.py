from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def huber_transform(x):
    x = x.clone()
    abs_x = torch.abs(x)
    x[abs_x < 1] = x[abs_x < 1].pow_(2).mul_(0.5)
    x[abs_x >= 1] = x[abs_x >= 1].abs_()
    return x

def quantized_loss(params, const):
    loss = 0
    for p in params:
        loss = loss + const * (torch.abs(p) - 1).norm(p=1)
    return loss

class BinaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        grad_output[x.abs() > 1] = 0
        return grad_output * x

binarize = BinaryFunction.apply
