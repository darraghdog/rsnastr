from typing import Any

from pytorch_toolbelt import losses
from pytorch_toolbelt.losses import BinaryFocalLoss
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, device, alpha=1, gamma=2, reduce=True):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.bce = BCEWithLogitsLoss(reduction = 'none').to(device)

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def getLoss(ltype, device):
    if ltype == "BinaryCrossentropy":
        return BinaryCrossentropy().to(device)
    if ltype == "BCEWithLogitsLoss":
        return BCEWithLogitsLoss().to(device)
    if ltype == "FocalBCEWithLogitsLoss":
        return FocalBCEWithLogitsLoss(device).to(device)
    if ltype == "FocalLoss":
        return FocalLoss()

def groupBy(samples, labels, unique_labels, labels_count, grptype = 'mean'):
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    if grptype == 'sum':
        return res
    if grptype == 'mean':
        res = res / labels_count.float().unsqueeze(1)
        return res
    
def rsna_criterion_all(y_pred_exam_, 
                   y_true_exam_, 
                   y_pred_img_, 
                   y_true_img_,
                   le_study, 
                   img_wt):
    # Groupby 
    labels = le_study.view(le_study.size(0), 1).expand(-1, 1)
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    
    #logger.info('Exam loss')
    exam_loss = bce_func_exam(y_pred_exam_, y_true_exam_)
    exam_loss = exam_loss.sum(1).unsqueeze(1)
    exam_loss = groupBy(exam_loss, labels, unique_labels, labels_count, grptype = 'mean').sum()
    exam_wts = torch.tensor(le_study.unique().shape[0]).float()
    
    #logger.info('Image loss')
    image_loss = bce_func_img(y_pred_img_, y_true_img_)
    image_loss = groupBy(image_loss, labels, unique_labels, labels_count, grptype = 'sum')
    qi_all = groupBy(y_true_img_, labels, unique_labels, labels_count, grptype = 'mean')
    image_loss = (img_wt * qi_all * image_loss).sum()
    img_wts = (img_wt * y_true_img_).sum()
    
    #logger.info('Final loss')
    img_loss_out =  image_loss / img_wts
    exam_loss_out = exam_loss / exam_wts
    final_loss = (image_loss + exam_loss)/(img_wts + exam_wts)
    return final_loss , img_loss_out, exam_loss_out