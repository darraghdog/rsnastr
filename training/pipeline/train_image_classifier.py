# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import sys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensor
from collections import defaultdict, OrderedDict
import platform
PATH = '/Users/dhanley/Documents/kaggle/rsnastr' \
        if platform.system() == 'Darwin' else 'mount'
os.chdir(PATH)
sys.path.append(PATH)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss
from training.tools.utils import get_logger
from training.tools.config import load_config
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from training.datasets.dataset import RSNAClassifierDataset, valSeedSampler, collatefn, nSampler
from training.zoo import classifiers
from training.tools.utils import create_optimizer, AverageMeter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logger = get_logger('Train', 'INFO') 

logger.info('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=8, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default='weights/')
arg('--resume', type=str, default='')
arg('--fold', type=int, default=0)
arg('--accum', type=int, default=1)
arg('--batchsize', type=int, default=4)
arg('--labeltype', type=str, default='all') # or 'single'
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
arg('--label-smoothing', type=float, default=0.01)
args = parser.parse_args()

if False:
    args.config = 'configs/effnetb5_lr5e4_multi.json'
conf = load_config(args.config)
logger.info(conf)

# Try using imagenet means
def create_train_transforms(size=300, distort = False):
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, value = 0,
                             rotate_limit=20, p=0.5, border_mode = cv2.BORDER_CONSTANT),
        A.Resize(size, size, p=1),
        A.Normalize(mean=conf['normalize']['mean'],
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
        ])

def create_val_transforms(size=300, HFLIPVAL = 1.0, TRANSPOSEVAL = 1.0):
    return A.Compose([
        A.Normalize(mean=conf['normalize']['mean'], 
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])

logger.info('Create traindatasets')
trndataset = RSNAClassifierDataset(mode="train",\
                                       fold=args.fold,\
                                       imgsize = conf['size'],\
                                       crops_dir=args.crops_dir,\
                                       imgclasses=conf["image_target_cols"],\
                                       studyclasses=conf['exam_target_cols'],\
                                       data_path=args.data_dir,\
                                       label_smoothing=args.label_smoothing,\
                                       folds_csv=args.folds_csv,\
                                       transforms=create_train_transforms(conf['size']))
logger.info('Create valdatasets')
valdataset = RSNAClassifierDataset(mode="valid",
                                    fold=args.fold,
                                    crops_dir=args.crops_dir,
                                    imgclasses=conf["image_target_cols"],
                                    studyclasses=conf['exam_target_cols'],
                                    imgsize = conf['size'],
                                    data_path=args.data_dir,
                                    folds_csv=args.folds_csv,
                                    transforms=create_val_transforms(conf['size']))

examlevel =  False # len(conf['exam_weights']) > 0
logger.info(f"Use {'EXAM' if examlevel else 'IMAGE'} level valid sampler")
valsampler = nSampler(valdataset.data, 
                          examlevel = examlevel,
                          pe_weight = conf['pe_ratio'], 
                          nmin = conf['studynmin'], 
                          nmax = conf['studynmax'], 
                          seed = None)

loaderargs = {'num_workers' : args.workers, 
              'pin_memory': False, 
              'drop_last': False, 
              'collate_fn' : collatefn}
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)

logger.info('Create model and optimisers')
nclasses = len(conf["image_target_cols"]) + \
            len(conf['exam_target_cols'])
logger.info(f'Nclasses : {nclasses}')
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'],nclasses = nclasses)
model = model.to(args.device)

image_weight = conf['image_weight'] if 'image_weight' in conf else 1.
logger.info(f'Image BCE weight :{image_weight}')
exam_bce_wts = torch.tensor(conf['exam_weights']).to(args.device)
img_bce_wts = torch.tensor([image_weight]).to(args.device)
logger.info(f"Multi BCE weights :{conf['exam_weights']}")
imgcriterion = torch.nn.BCEWithLogitsLoss(reduction='mean', weight = img_bce_wts)
examcriterion = torch.nn.BCEWithLogitsLoss(reduction='none', weight = exam_bce_wts)

optimizer, scheduler = create_optimizer(conf['optimizer'], model)
bce_best = 100
batch_size = conf['optimizer']['batch_size']

if conf['fp16'] and args.device != 'cpu':
    scaler = torch.cuda.amp.GradScaler()
    
    
f"_epoch{current_epoch}"
    
snapshot_name = "{}_{}_{}_fold{}_img{}_accum{}_".format(conf.get("prefix", args.prefix), 
                                     conf['network'], 
                                     conf['encoder'], 
                                     args.fold,
                                     conf['size'],
                                     args.accum,
                                     )
max_epochs = conf['optimizer']['schedule']['epochs']

logger.info('Start training')
epoch_img_names = defaultdict(list)
seenratio=0  # Ratio of seen in images in previous epochs

current_epoch = 1
for epoch in range( max_epochs):
    
    '''
    TRAIN
    '''
    ep_samps={'tot':0,'pos':0}
    losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    tot_exam_loss = 0.
    tot_img_loss = 0.
    skipped =  0
    logger.info(f"Use {'EXAM' if examlevel else 'IMAGE'} level train sampler")
    trnsampler = nSampler(trndataset.data, 
                          examlevel = examlevel,
                          pe_weight = conf['pe_ratio'], 
                          nmin = conf['studynmin'], 
                          nmax = conf['studynmax'], 
                          seed = None)
        
    trncts = trndataset.data.iloc[trnsampler.sample(trndataset.data)].pe_present_on_image.value_counts()
    valcts = valdataset.data.iloc[valsampler.sample(trndataset.data)].pe_present_on_image.value_counts()
    logger.info(f'Train class balance:\n{trncts}')
    logger.info(f'Valid class balance:\n{valcts}')
    trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = trnsampler, **loaderargs)
    model.train()
    pbar = tqdm(enumerate(trnloader), total=len(trnloader), desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "current_epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        epoch_img_names[current_epoch] += sample['img_name']
        imgs = sample["image"].to(args.device)
        # logger.info(f'Mean {imgs.mean()} std {imgs.std()} ')
        labels = sample["labels"].to(args.device).float()
        mask = sample["labels"][:,0]
        if args.device != 'cpu':
            with autocast():
                out = model(imgs)
        else:
            out = model(imgs)
        imgloss = imgcriterion(out[:,:1], labels[:,:1]) 
        examloss = examcriterion(out[:,1:], labels[:,1:]) 
        # Mask the loss of the multi classes
        if sum(mask>=0.5)>0:
            examloss = (examloss.sum(1)[mask>=0.5]).mean()
            loss = imgloss + examloss
        else:
            skipped += 1 
            del imgloss, examloss, out
            continue
        if args.device != 'cpu':
            scaler.scale(loss).backward()
            if (i % args.accum) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        i -= skipped
        losses.update(loss.item(), imgs.size(0))
        tot_img_loss += imgloss.item()
        tot_exam_loss += examloss.item()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, 
                          "loss": losses.avg, 
                          "loss_exam": tot_exam_loss / (i+1), 
                          "loss_img": tot_img_loss / (i+1), 
                          'seen_prev': seenratio })
        
        del imgs, labels, mask
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i%5==0:
            
            torch.cuda.empty_cache()
            # Update the amount of images we have seen already
            seen = set(epoch_img_names[epoch]).intersection(
                set(itertools.chain(*[epoch_img_names[i] for i in range(epoch)])))
            seenratio = len(seen)/len(epoch_img_names[epoch])
    pbar.close()

    
    '''
    Validate
    '''
    model = model.eval()
    tot_exam_loss = 0.
    tot_img_loss = 0.
    skipped = 0
    losses = AverageMeter()
    pbarval = tqdm(enumerate(valloader), total=len(valloader), desc="Epoch valid {}".format(current_epoch), ncols=0)
    for i, sample in pbarval:
        imgs = sample["image"].to(args.device)
        labels = sample["labels"].to(args.device).float()
        mask = sample["labels"][:,0]
        with torch.no_grad():
            out = model(imgs)
            imgloss = imgcriterion(out[:,:1], labels[:,:1]) 
            examloss = examcriterion(out[:,1:], labels[:,1:]) 
            # Mask the loss of the multi classes
            if sum(mask>=0.5)>0:
                examloss = (examloss.sum(1)[mask>=0.5]).mean()
                loss = imgloss + examloss
            else:
                skipped += 1
                del imgloss, examloss, out
                continue
        i -= skipped
        losses.update(loss.item(), imgs.size(0))
        tot_img_loss += imgloss.item()
        if sum(mask>=0.5)>0: tot_exam_loss += examloss.item()
        pbarval.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, 
                          "loss": losses.avg, 
                          "loss_exam": tot_exam_loss / (i+1), 
                          "loss_img": tot_img_loss / (i+1)})
        if i%5==0: 
            del imgs, labels, mask
            torch.cuda.empty_cache()

    '''
    Save the model
    '''
    bce = losses.avg
    
    if bce < bce_best:
        bce_best = bce
        wtname = conf['weights'] if 'weights' in conf else ''
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce,
            }, args.output_dir + snapshot_name + f"__best")

    print("Epoch: {} bce: {:.5f}, bce_best: {:.5f}".format(current_epoch, bce, bce_best))
    current_epoch += 1

