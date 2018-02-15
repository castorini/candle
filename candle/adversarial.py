from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import SerializableModule

class AdversarialCriterion(object):
    def __init__(self, model_a, model_b, ce_margin=0.1, ce_lambda=1):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.ce_margin = ce_margin
        self.ce_lambda = ce_lambda

    def __call__(self, loss1, loss2, loss_model=None):
        ce_margin_loss1 = self.ce_lambda * F.relu(self.ce_margin + loss1 - loss2)
        ce_margin_loss2 = self.ce_lambda * F.relu(self.ce_margin + loss2 - loss1)
        if loss_model is None:
            loss = ce_margin_loss1 + ce_margin_loss2
        elif loss_model is self.model_a:
            loss = ce_margin_loss1
        elif loss_model is self.model_b:
            loss = ce_margin_loss2
        return loss
