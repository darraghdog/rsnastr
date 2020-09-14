# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
from collections import defaultdict, OrderedDict
import platform
PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
os.chdir(PATH)

from sklearn.metrics import log_loss
from utils.logs import get_logger
from utils.utils import RSNAWEIGHTS
from training.tools.config import load_config
import pandas as pd
import cv2


import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from training.datasets.classifier_dataset import RSNAClassifierDataset, nSampler
from training.zoo import classifiers


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
logger = get_logger('Train', 'INFO') 

# Data loaders
mean_img = [0.22363983, 0.18190407, 0.2523437 ]
std_img = [0.32451536, 0.2956294,  0.31335256]

def create_train_transforms(size=300):
    return A.Compose([
        #A.HorizontalFlip(p=0.5),   # right/left
        A.VerticalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                             rotate_limit=20, p=0.5, border_mode = cv2.BORDER_REPLICATE),
        A.Cutout(num_holes=40, max_h_size=size//7, max_w_size=size//7, fill_value=128, p=0.5), 
        #A.Transpose(p=0.5), # swing in -90 degrees
        A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])

def create_val_transforms(size=300, HFLIPVAL = 1.0, TRANSPOSEVAL = 1.0):
    return A.Compose([
        #A.HorizontalFlip(p=HFLIPVAL),
        #A.Transpose(p=TRANSPOSEVAL),
        A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])

parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=6, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'gpu', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default='weights/')
arg('--resume', type=str, default='')
arg('--fold', type=int, default=0)
arg('--batchsize', type=int, default=8)
arg('--imgsize', type=int, default=512)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpeg')
arg('--label-smoothing', type=float, default=0.01)
arg('--logdir', type=str, default='logs')
arg('--distributed', action='store_true', default=False)
arg('--freeze-epochs', type=int, default=0)
arg('--size', type=int, default=300)
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
arg("--opt-level", default='O1', type=str)
arg("--test_every", type=int, default=1)
args = parser.parse_args()



trndataset = RSNAClassifierDataset(mode="train",
                                       fold=args.fold,
                                       imgsize = args.imgsize,
                                       crops_dir=args.crops_dir,
                                       data_path=args.data_dir,
                                       label_smoothing=args.label_smoothing,
                                       folds_csv=args.folds_csv,
                                       transforms=create_train_transforms(args.imgsize))
valdataset = RSNAClassifierDataset(mode="val",
                                     fold=args.fold,
                                     crops_dir=args.crops_dir,
                                     imgsize = args.imgsize,
                                     data_path=args.data_dir,
                                     folds_csv=args.folds_csv,
                                     transforms=create_val_transforms(args.imgsize))
trnsampler = nSampler(trndataset.data, 4)
valsampler = nSampler(valdataset.data, 4)
loaderargs = {'num_workers' : 8}#, 'collate_fn' : collatefn}
trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = trnsampler, **loaderargs)
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)

args.config = 'configs/b5.json'
conf = load_config(args.config)
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'])

model = model.to(args.device)
reduction = "mean"


'''
Set up the losses so that if the image level does not have `pe_present_on_image`
then we mark it as negative exam for pe. and all the labels are 0 ???????
'''

    loss_fn = []
    weights = []
    for loss_name, weight in conf["losses"].items():
        loss_fn.append(losses.__dict__[loss_name](reduction=reduction).cuda())
        weights.append(weight)

#%time for i in range(1000) : a = next(iter(data_train))



