# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import glob
import pickle
import gc
import sys
import itertools
from collections import defaultdict, OrderedDict
import platform
PATH = '/Users/dhanley/Documents/rsnastr'         if platform.system() == 'Darwin' else '/mount'
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

from tqdm import tqdm
import torch.distributed as dist
from training.datasets.classifier_dataset import RSNAImageSequenceDataset, collateseqimgfn
from training.zoo.sequence import StudyImgNet
from training.tools.utils import create_optimizer, AverageMeter, collectPreds, collectLoss
from training.tools.utils import splitbatch, unmasklabels, unmasklogits
from training.losses import getLoss
from training import losses
from torch.optim.swa_utils import AveragedModel, SWALR
from tensorboardX import SummaryWriter
from torch.utils.data import WeightedRandomSampler

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
logger = get_logger('LSTM', 'INFO') 


# In[2]:


def create_train_transforms_multi(size=300, distort = False):
    return A.Compose([
        #A.HorizontalFlip(p=0.5),   # right/left
        #A.VerticalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, value = 0,
                                 rotate_limit=10, p=0.5, border_mode = cv2.BORDER_CONSTANT),
        # A.Cutout(num_holes=40, max_h_size=size//7, max_w_size=size//7, fill_value=128, p=0.5), 
        #A.Transpose(p=0.5), # swing in -90 degrees
        A.Resize(size, size, p=1), 
        A.Normalize(mean=conf['normalize']['mean'], 
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
        ])

def create_val_transforms(size=300, HFLIPVAL = 1.0, TRANSPOSEVAL = 1.0):
    return A.Compose([
        #A.HorizontalFlip(p=HFLIPVAL),
        #A.Transpose(p=TRANSPOSEVAL),
        A.Normalize(mean=conf['normalize']['mean'], 
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])


# In[3]:

'''
import sys; sys.argv=['']; del sys
'''
logger.info('Load args')

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=6, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default='weights/')
arg('--resume', type=str, default='')
arg('--fold', type=int, default=0)
arg('--batchsize', type=int, default=1)
arg('--lr', type=float, default = 0.0001)
arg('--lrgamma', type=float, default = 0.98)
arg('--labeltype', type=str, default='all') # or 'single'
arg('--dropout', type=float, default = 0.2)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--nclasses', type=str, default=1)
arg('--crops-dir', type=str, default='jpegip')
arg('--lstm_units',   type=int, default=512)
arg('--epochs',   type=int, default=32)
arg('--nbags',   type=int, default=12)
arg('--accum', type=int, default=16)
arg('--label-smoothing', type=float, default=0.00)
arg('--logdir', type=str, default='logs/b2_1820')
arg("--local_rank", default=0, type=int)
arg('--embrgx', type=str, default='weights/image_weights_regex')
arg("--seed", default=777, type=int)
args = parser.parse_args()

# In[4]:
conf = load_config(args.config)
# In[5]:

# In[6]:


logger.info('Create traindatasets')
trndataset = RSNAImageSequenceDataset(mode="train",                                       fold=args.fold,                                       pos_sample_weight = conf['pos_sample_weight'],                                       sample_count = conf['sample_count'],                                        imgsize = conf['size'],                                       crops_dir=args.crops_dir,                                       balanced=conf['balanced'],                                       imgclasses=conf["image_target_cols"],                                       studyclasses=conf['exam_target_cols'],                                       data_path=args.data_dir,                                       label_smoothing=args.label_smoothing,                                       folds_csv=args.folds_csv,                                       transforms=create_train_transforms_multi(conf['size'])                                           if len(conf['exam_target_cols'])>0 else                                            create_train_transforms_binary(conf['size']))
logger.info('Create valdatasets')
valdataset = RSNAImageSequenceDataset(mode="valid",
                                    fold=args.fold,
                                    pos_sample_weight = conf['pos_sample_weight'],
                                    sample_count = conf['sample_count'], 
                                    crops_dir=args.crops_dir,
                                    balanced=conf['balanced'],
                                    imgclasses=conf["image_target_cols"],
                                    studyclasses=conf['exam_target_cols'],
                                    imgsize = conf['size'],
                                    data_path=args.data_dir,
                                    folds_csv=args.folds_csv,
                                    transforms=create_val_transforms(conf['size']))


# In[7]:


logger.info('Create loaders...')
def sampler(dataset):
    wts = dataset.folddf.negative_exam_for_pe.values
    w0 = (wts>0.5).sum()
    w1 = (wts<0.5).sum()
    wts[wts==1] = w1
    wts[wts==0] = w0
    sampler = WeightedRandomSampler(wts, len(wts), replacement=True)
    return sampler
    
valloader = DataLoader(valdataset, 
                       batch_size=args.batchsize, 
                       shuffle=False,
                       # sampler=sampler(valdataset), 
                       num_workers=16, 
                       collate_fn=collateseqimgfn)
# del embmat
gc.collect()

logger.info('Create model')
nc = len(conf['image_target_cols']+conf['exam_target_cols'])
model =StudyImgNet(conf['encoder'], 
                   dropout = 0.2,
                   nclasses = nc,
                   dense_units = 512)
'''
batch = next(iter(trnloader))
x = batch['image']
out = model(x)
'''
model = model.to(args.device)
DECAY = 0.0
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = torch.optim.Adam(plist, lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lrgamma, last_epoch=-1)

# Exam Loss
bcewLL_func = torch.nn.BCEWithLogitsLoss(reduction='none')


# In[8]:


ypredls = []
ypredtstls = []
scaler = torch.cuda.amp.GradScaler()


# In[ ]:


logger.info('Start training')
for epoch in range(args.epochs):
    trnloader = DataLoader(trndataset, 
                       batch_size=args.batchsize, 
                       sampler=sampler(trndataset), 
                       num_workers=16, 
                       collate_fn=collateseqimgfn)
    logger.info(50*'-')
    trnloss   = 0.
    trnwts    = 0.
    trnimgloss   = 0.
    trnimgwts    = 0.
    trnexmloss   = 0.
    trnexmwts    = 0.
    label_w = torch.tensor(conf['exam_weights']).to(args.device, dtype=torch.float)
    img_w = torch.tensor(conf['image_weight']).to(args.device, dtype=torch.float)
    model = model.train()
    pbar = tqdm(enumerate(trnloader), 
                total = len(trndataset)//trnloader.batch_size, 
                desc=f"Train epoch {epoch}", ncols=0)
    for step, batch in pbar:
        ytrn = batch['labels'].to(args.device, dtype=torch.float)
        xtrn = batch['image'].to(args.device, dtype=torch.float)
        xtrn = torch.autograd.Variable(xtrn, requires_grad=True)
        ytrn = torch.autograd.Variable(ytrn)
        ytrn = ytrn.view(-1, 10)
        #logger.info(xtrn.shape)
        with autocast():
            outimg, outexm = model(xtrn)
            # Exam loss
            exam_loss = bcewLL_func(outexm, ytrn[:1,1:])
            exam_loss = torch.sum(exam_loss*label_w, 1)[0]
            # Image loss
            y_pred_img_ = outimg.squeeze(-1)
            y_true_img_ = ytrn[:,:1].transpose(0,1)
            image_loss = bcewLL_func(y_pred_img_, y_true_img_)
            img_num = y_pred_img_.shape[-1]
            qi = torch.sum(y_true_img_)
            image_loss = torch.sum(img_w*qi*image_loss)
            # Sum it all
            samploss = exam_loss+image_loss
            sampwts = label_w.sum() + (img_w*qi*img_num)
        loss = (samploss/sampwts) / args.accum
        scaler.scale(loss).backward()
        trnloss += samploss.item()
        trnwts += sampwts.item()
        trnimgloss   += image_loss.item()
        trnimgwts    += (img_w*qi*img_num).item()
        trnexmloss   += exam_loss.item()
        trnexmwts    += label_w.sum().item()
        # logger.info(f'{image_loss.item():.4f}\t{(img_w*qi*img_num).item():.4f}\t{exam_loss.item():.4f}\t{label_w.sum().item():.4f}\t')
        if (step+1) % args.accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        final_trn_loss = trnloss/trnwts
        pbar.set_postfix({'train loss': final_trn_loss, 
                          'image loss': (trnimgloss/trnimgwts) if trnimgloss>0 else 0, 
                          'exam loss': trnexmloss/trnexmwts})
        del xtrn, ytrn, outimg, outexm
        if step%100==0:
            torch.cuda.empty_cache()    
    logger.info(f'Epoch {epoch} train loss all {final_trn_loss:.4f}')
    output_model_file = f'weights/exam_lstm_{conf["encoder"]}_epoch{epoch}_fold{args.fold}.bin'
    torch.save(model.state_dict(), output_model_file)
    scheduler.step()
    model.eval()
    valloss   = 0.
    valwts    = 0.
    valimgloss   = 0.
    valimgwts    = 0.
    valexmloss   = 0.
    valexmwts    = 0.
    ypredls = []
    yvalls = []
    pbarval = tqdm(enumerate(valloader), 
                   total = len(valdataset)//valloader.batch_size, 
                   desc="Train epoch {}".format(epoch), ncols=0)
    for step, batch in pbarval:
        y = batch['labels'].to(args.device, dtype=torch.float)
        x = batch['image'].to(args.device, dtype=torch.float)
        #logger.info(xval.shape)
        y = y.view(-1, 10)
        with torch.no_grad():
            outimg, outexm = model(x)
            # Exam loss
            exam_loss = bcewLL_func(outexm, y[:1,1:])
            exam_loss = torch.sum(exam_loss*label_w, 1)[0]
            # Image loss
            y_pred_img_ = outimg.squeeze(-1)
            y_true_img_ = y[:,:1].transpose(0,1)
            image_loss = bcewLL_func(y_pred_img_, y_true_img_)
            img_num = y_pred_img_.shape[-1]
            qi = torch.sum(y_true_img_)
            image_loss = torch.sum(img_w*qi*image_loss)
            # Sum it all
            samploss = exam_loss+image_loss
            sampwts = label_w.sum() + img_w*qi*img_num
        valloss += samploss.item()
        valwts += sampwts.item()
        valimgloss   += image_loss.item()
        valimgwts    += (img_w*qi*img_num).item()
        valexmloss   += exam_loss.item()
        valexmwts    += label_w.sum().item()
        final_val_loss = valloss/valwts
        pbar.set_postfix({'valid loss': final_trn_loss,
                          'image loss': (valimgloss/trnimgwts) if valimgloss>0 else 0,
                          'exam loss': valexmloss/valexmwts})
        del x, y, outimg, outexm
        torch.cuda.empty_cache()
    logger.info(f'Epoch {epoch} valid loss all {final_val_loss:.4f}')


# In[ ]:




