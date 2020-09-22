from typing import Any

from pytorch_toolbelt import losses
from pytorch_toolbelt.losses import BinaryFocalLoss
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

class WeightedLosses(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, *input: Any, **kwargs: Any):
        cum_loss = 0
        for loss, w in zip(self.losses, self.weights):
            cum_loss += w * loss.forward(*input, **kwargs)
        return cum_loss
    
class BinaryCrossentropy(BCEWithLogitsLoss):
    pass

class FocalLoss(BinaryFocalLoss):
    def __init__(self, alpha=None, gamma=3, ignore_index=None, reduction="mean", normalized=False,
                 reduced_threshold=None):
        super().__init__(alpha, gamma, ignore_index, reduction, normalized, reduced_threshold)
        
        
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def getLoss(ltype, weights):
    if ltype == "BinaryCrossentropy":
        return BinaryCrossentropy(weights)
    if ltype == "BCEWithLogitsLoss":
        return BCEWithLogitsLoss(weights)
    if ltype == "FocalBCEWithLogitsLoss":
        return FocalBCEWithLogitsLoss(weights)
    if ltype == "FocalLoss":
        return FocalLoss(weights)
