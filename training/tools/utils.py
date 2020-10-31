# https://github.com/ildoonet/pytorch-gradual-warmup-lr
# https://github.com/PavelOstyakov/predictions_balancing/blob/master/run.py
import pickle
import argparse
import os
import torch
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict, OrderedDict
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import platform
import os
import gc
import glob
import pydicom
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import sys
import logging
import cv2
#from apex.optimizers import FusedAdam, FusedSGD
import numpy as np
from timm.optim import AdamW
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, StepLR

from training.tools.schedulers import ExponentialLRScheduler, PolyLR, LRStepScheduler

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

RSNA_CFG = {
    'image_target_cols': [
        'pe_present_on_image', # only image level
    ],
    'exam_target_cols': [
        'negative_exam_for_pe', # exam level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ], 
    'image_weight': 0.07361963,
    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988],
}

class resultsfn:
    loss   = 0.
    wts    = 0.
    imgloss   = 0.
    imgwts    = 0.
    exmloss   = 0.
    exmwts    = 0.

def get_level(level_str):
    ''' get level'''
    l_names = {logging.getLevelName(lvl).lower(): lvl for lvl in [10, 20, 30, 40, 50]} # noqa
    return l_names.get(level_str.lower(), logging.INFO)

def get_logger(name, level_str):
    ''' get logger'''
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level_str))
    handler = logging.StreamHandler()
    handler.setLevel(level_str)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')) # pylint: disable=C0301 # noqa
    logger.addHandler(handler)

    return logger

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def turbodump(f, img):
    # encoding BGR array to output.jpg with default settings.
    out_file = open(f, 'wb')
    out_file.write(jpeg.encode(img[:,:,::-1]))
    out_file.close()
    
# decoding input.jpg to BGR array
def turboload(f):
    in_file = open(f, 'rb')
    bgr_array = jpeg.decode(in_file.read())
    in_file.close()
    return bgr_array[:,:,::-1]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(optimizer_config, model, master_params=None):
    """Creates optimizer and schedule from configuration
    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.
    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config.get("classifier_lr", -1) != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("encoder") != -1:
                net_params.append(v)
            else:
                classifier_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        if master_params:
            params = master_params
        else:
            params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "FusedSGD":
        optimizer = FusedSGD(params,
                             lr=optimizer_config["learning_rate"],
                             momentum=optimizer_config["momentum"],
                             weight_decay=optimizer_config["weight_decay"],
                             nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"])
    elif optimizer_config["type"] == "FusedAdam":
        optimizer = FusedAdam(params,
                              lr=optimizer_config["learning_rate"],
                              weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "AdamW":
        optimizer = AdamW(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "RmsProp":
        optimizer = RMSprop(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))


    if optimizer_config["schedule"]["type"] == "step":
        scheduler = LRStepScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "steplr":
        scheduler = StepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = ExponentialLRScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]
        
        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler

def unmasklabels(yimg, ystudy, lelabels, img_names, mask):
    ystudy = ystudy.unsqueeze(2).repeat(1, 1, yimg.size(1))
    ystudy = ystudy.transpose(2, 1)
    # get the mask for masked img labels
    maskidx = mask.view(-1)==1
    # Flatten them all along batch and seq dimension and remove masked values
    yimg = yimg.view(-1, 1)[maskidx]
    ystudy = ystudy.reshape(-1, ystudy.size(-1))[maskidx]
    lelabels = lelabels.view(-1, 1)[maskidx] 
    lelabels = lelabels.flatten()
    img_names = img_names.flatten()[maskidx.detach().cpu().numpy()]
    return yimg, ystudy, lelabels, img_names

def unmasklogits(imglogits, studylogits, mask):
    imglogits = imglogits.squeeze()
    studylogits = studylogits.unsqueeze(2).repeat(1, 1, imglogits.size(1))
    # get the mask for masked img labels
    maskidx = mask.view(-1)==1
    # Flatten them all along batch and seq dimension and remove masked values
    imglogits = imglogits.view(-1, 1)[maskidx]
    #studylogits = studylogits.reshape(-1, ystudy.size(-1))[maskidx]
    studylogits = studylogits.reshape(-1, studylogits.size(1))[maskidx]
    return imglogits, studylogits


def splitbatch(batch, device):
    img_names = batch['img_name']
    yimg = batch['imglabels'].to(device, dtype=torch.float)
    ystudy = batch['studylabels'].to(device, dtype=torch.float)
    mask = batch['mask'].to(device, dtype=torch.int)
    lelabels = batch['lelabels'].to(device, dtype=torch.int64)
    return img_names, yimg, ystudy, mask, lelabels