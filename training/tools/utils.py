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


class collectPreds:
    def __init__(self):
        self.lelabelsls = []
        self.imgnamesls = []
        self.imgpredsls = []
        self.imglabells = []
        self.studylabells = []
        self.studypredsls = []
        self.maxlelabel = 0

    def append(self, img_names, lelabels, imgpreds, studypreds, yimg, ystudy):
        lelabels = lelabels.detach().cpu()
        if len(self.lelabelsls)>0:
            increment = self.lelabelsls[-1].max() + torch.tensor(1).cpu()
            lelabels = lelabels + increment
        self.lelabelsls.append(lelabels)
        self.imgpredsls.append(imgpreds.detach().cpu())
        self.imglabells.append(yimg.detach().cpu())
        self.studylabells.append(ystudy.detach().cpu())
        self.studypredsls.append(studypreds.detach().cpu())
        self.imgnamesls.append(img_names)

    def concat(self, device):
        lelabels = torch.cat(self.lelabelsls).to(device)
        imgpreds = torch.cat(self.imgpredsls).to(device)
        imglabels = torch.cat(self.imglabells).to(device)
        studylabels = torch.cat(self.studylabells).to(device)
        studypreds = torch.cat(self.studypredsls).to(device)
        return studypreds, studylabels, imgpreds, imglabels, lelabels
    
    def series(self, series):
        if series=='lelabels': return torch.cat(self.lelabelsls)
        if series=='img_preds': return torch.cat(self.imgpredsls)
        if series=='img_labels': return torch.cat(self.imglabells)
        if series=='study_labels': return torch.cat(self.studylabells)
        if series=='study_preds': return torch.cat(self.studypredsls)
        if series=='img_names': return np.concatenate(self.imgnamesls)

class collectLoss:
    def __init__(self, loader, mode = 'train'):
        self.mode = mode
        self.loss = 0.
        self.img_loss = 0.
        self.exam_loss = 0.
        self.step = 1
        self.loaderlen = len(loader)

    def increment(self, loss, img_loss, exam_loss):
        self.loss += loss.item()
        self.img_loss += img_loss.item()
        self.exam_loss += exam_loss.item()
        self.step += 1

    def log(self):
        logs = f'{self.mode} step {self.step} of {self.loaderlen} trn loss {(self.loss/(self.step)):.4f} '
        logs += f'img loss {(self.img_loss/(self.step)):.4f} exam loss {(self.exam_loss/(self.step)):.4f}'
        return logs

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