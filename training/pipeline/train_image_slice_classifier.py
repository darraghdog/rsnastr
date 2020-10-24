# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import sys
import itertools
from collections import defaultdict, OrderedDict
import platform
PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/mount'
os.chdir(PATH)
sys.path.append(PATH)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss
from utils.logs import get_logger
from utils.utils import RSNAWEIGHTS, RSNA_CFG as CFG
from training.tools.config import load_config
import pandas as pd
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import random

from tqdm import tqdm
import torch.distributed as dist
from training.datasets.classifier_dataset import RSNAClassifierDataset, \
        nSampler, valSeedSampler, collatefn, RSNASliceClassifierDataset
from training.zoo import classifiers
from training.zoo.classifiers import validate
from training.tools.utils import create_optimizer, AverageMeter
from training.losses import getLoss
from training import losses

from tensorboardX import SummaryWriter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
logger = get_logger('Train', 'INFO') 


logger.info('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=6, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default='weights/')
arg('--resume', type=str, default='')
arg('--fold', type=int, default=0)
arg('--accum', type=int, default=1)
arg('--batchsize', type=int, default=4)
arg('--labeltype', type=str, default='all') # or 'single'
arg('--augextra', type=str, default=False) # or 'single'
arg('--mixup_beta', type=float, default = 0.)
arg('--step', type=float, default = 1)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
arg('--label-smoothing', type=float, default=0.01)
arg('--logdir', type=str, default='logs/b2_1820')
arg('--distributed', action='store_true', default=False)
arg('--freeze-epochs', type=int, default=0)
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
arg("--opt-level", default='O1', type=str)
arg("--test_every", type=int, default=1)
arg('--from-zero', action='store_true', default=False)
arg('--flip', type=str, default=False)
args = parser.parse_args()
args.flip = args.flip=='True'

if False:
    args.config = 'configs/b2.json'
    args.config = 'configs/b2_binary.json'
    args.config = 'configs/rnxt101_binary.json'
    args.config = 'configs/512/effnetb5_lr5e4_multi.json'
conf = load_config(args.config)
logger.info(conf)

# Try using imagenet means
def create_train_transforms():
    rot = random.randrange(-20, 20)
    ss1 = random.randrange(0, 5) / 100
    ss2 = random.randrange(0, 5) / 100
    return A.Compose([
        A.ShiftScaleRotate(p=1.0, rotate_limit=(rot,rot),
                           shift_limit=ss1, scale_limit=ss2,
                           border_mode = cv2.BORDER_CONSTANT),
        A.Normalize(mean=conf['normalize']['mean'],
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
        ])

def create_val_transforms(size=300):
    return A.Compose([
        A.Normalize(mean=conf['normalize']['mean'],
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
        ])

logger.info('Create traindatasets')
trndataset = RSNASliceClassifierDataset(mode="train",\
                                       flip=args.flip,\
                                       fold=args.fold,\
                                       step=args.step,\
                                       imgsize = conf['size'],\
                                       crops_dir=args.crops_dir,\
                                       imgclasses=conf["image_target_cols"],\
                                       studyclasses=conf['exam_target_cols'],\
                                       data_path=args.data_dir,\
                                       label_smoothing=args.label_smoothing,\
                                       folds_csv=args.folds_csv,\
                                       transforms=create_train_transforms)
logger.info('Create valdatasets')
valdataset = RSNASliceClassifierDataset(mode="valid",
                                    fold=args.fold,
                                    step=args.step,
                                    crops_dir=args.crops_dir,
                                    imgclasses=conf["image_target_cols"],
                                    studyclasses=conf['exam_target_cols'],
                                    imgsize = conf['size'],
                                    data_path=args.data_dir,
                                    folds_csv=args.folds_csv,
                                    transforms=create_val_transforms)

examlevel =  False # len(conf['exam_weights']) > 0
logger.info(f"Use {'EXAM' if examlevel else 'IMAGE'} level valid sampler")
valsampler = nSampler(valdataset.data, 
                          examlevel = examlevel,
                          pe_weight = conf['pe_ratio'], 
                          nmin = conf['studynmin'], 
                          nmax = conf['studynmax'], 
                          seed = None)

logger.info(50*'-')
loaderargs = {'num_workers' : 8, 'pin_memory': False, 'drop_last': False, 'collate_fn' : collatefn}
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)

logger.info('Create model and optimisers')
nclasses = len(conf["image_target_cols"]) + len(conf['exam_target_cols'])
logger.info(f'Nclasses : {nclasses}')
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'],nclasses = nclasses)
model = model.to(args.device)
num_channels = 9
logger.info(f'change num channels: {num_channels}')
model.encoder.conv_stem = nn.Conv2d(num_channels, 48, 3, 2, 1, bias = False)


image_weight = conf['image_weight'] if 'image_weight' in conf else 1.
logger.info(f'Image BCE weight :{image_weight}')
exam_bce_wts = torch.tensor(conf['exam_weights']).to(args.device)
img_bce_wts = torch.tensor([image_weight]).to(args.device)
logger.info(f"Multi BCE weights :{conf['exam_weights']}")
imgcriterion = torch.nn.BCEWithLogitsLoss(reduction='mean', weight = img_bce_wts)
examcriterion = torch.nn.BCEWithLogitsLoss(reduction='none', weight = exam_bce_wts)

optimizer, scheduler = create_optimizer(conf['optimizer'], model)
bce_best = 100
start_epoch = 0
batch_size = conf['optimizer']['batch_size']

os.makedirs(args.logdir, exist_ok=True)
summary_writer = SummaryWriter(args.logdir + '/' + conf.get("prefix", args.prefix) + conf['encoder'] + "_" + str(args.fold))

if args.from_zero:
    start_epoch = 0
current_epoch = start_epoch

if conf['fp16'] and args.device != 'cpu':
    scaler = torch.cuda.amp.GradScaler()
    
snapshot_name = "{}{}_{}_{}_".format(conf.get("prefix", args.prefix), conf['network'], conf['encoder'], args.fold)
max_epochs = conf['optimizer']['schedule']['epochs']

logger.info('Start training')
epoch_img_names = defaultdict(list)
seenratio=0  # Ratio of seen in images in previous epochs


for epoch in range(start_epoch, max_epochs):
    '''
    Here we took out a load of things, check back 
    https://github.com/selimsef/dfdc_deepfake_challenge/blob/9925d95bc5d6545f462cbfb6e9f37c69fa07fde3/training/pipelines/train_classifier.py#L188-L201
    '''
    
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
        if conf['fp16'] and args.device != 'cpu':
            with autocast():
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
                    
            scaler.scale(loss).backward()
            if (i % args.accum) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
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
        
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i%5==0:
            del imgs, labels, mask
            torch.cuda.empty_cache()

    pbar.close()
    if epoch > 0:
        seen = set(epoch_img_names[epoch]).intersection(
            set(itertools.chain(*[epoch_img_names[i] for i in range(epoch)])))
        seenratio = len(seen)/len(epoch_img_names[epoch])

    for idx, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)
    model = model.eval()
    # bce, acc, probdf = validate(model, valloader, device = args.device, logger = logger, half = False)
    '''
    Validate
    '''
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
    print("Epoch: {} bce: {:.5f}, bce_best: {:.5f}".format(current_epoch, bce, bce_best))
    torch.save({
        'epoch': current_epoch + 1,
        'state_dict': model.state_dict(),
        'bce_best': bce,
        }, args.output_dir + snapshot_name + f"_nclasses{nclasses}_size{conf['size']}_accum{args.accum}_slicestep{args.step}_fold{args.fold}_epoch{current_epoch}")
    current_epoch += 1