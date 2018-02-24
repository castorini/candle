import torch

from .model import SerializableModule

class HardRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stochastic):
        if stochastic:
            return torch.bernoulli(x.clamp(0, 1))
        return x.clamp(0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SmoothRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stochastic):
        ctx.save_for_backward(x)
        if stochastic:
            return torch.bernoulli(x.clamp(0, 1))
        return x.clamp(0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        sig = (6 * (x - 0.5)).sigmoid()
        return grad_output * 6 * sig * (1 - sig), None

class BinaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        grad_output[x.abs() > 1] = 0
        return grad_output

class BinaryTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 2 * ((x + 1) / 2).clamp(0, 1).round() - 1

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        grad_output[x.abs() > 1] = 0
        return grad_output

class ApproxPow2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ap2(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        return grad_output

binarize = BinaryFunction.apply
binary_tanh = BinaryTanhFunction.apply
approx_pow2 = ApproxPow2Function.apply
mask_round = HardRoundFunction.apply
smooth_round = SmoothRoundFunction.apply

class BinaryTanh(SerializableModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return binary_tanh(x)

class HardRound(SerializableModule):
    def __init__(self, stochastic=False):
        super().__init__()
        self.stochastic = stochastic

    def forward(self, x):
        return mask_round(x, self.stochastic)

class SmoothRound(SerializableModule):
    def __init__(self, stochastic=False):
        super().__init__()
        self.stochastic = stochastic

    def forward(self, x):
        return smooth_round(x, self.stochastic)