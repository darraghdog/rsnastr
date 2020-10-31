import argparse
import json
import os
import glob
import pickle
import gc
import sys
import itertools
from collections import defaultdict, OrderedDict, namedtuple
import platform
PATH = '/Users/dhanley/Documents/kaggle/rsnastr' \
        if platform.system() == 'Darwin' else '/mount'
sys.path.append(PATH)
os.chdir(PATH)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss
import pandas as pd
import random
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from tqdm import tqdm
import torch.distributed as dist
from training.datasets.dataset import RSNASequenceDataset, collateseqfn, valSeedSampler, examSampler
from training.zoo.sequence import SpatialDropout, LSTMNet, TransformerNet
from training.tools.utils import create_optimizer, AverageMeter
from training.tools.utils import splitbatch, unmasklabels, unmasklogits
from training.tools.utils  import get_logger, resultsfn
from training.tools.config import load_config, RSNA_CFG
from torch.optim.swa_utils import AveragedModel, SWALR

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
logger = get_logger('LSTM', 'INFO') 

logger.info('Load args')
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=6, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--output-dir', type=str, default='weights/')
arg('--fold', type=int, default=0)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg("--imgemb", type=str, default='weights.emb')
args = parser.parse_args()


if False:
    args.config = 'configs/b5_seq_lstm.json'
    args.config = 'configs/b5_seq_transformer_1layer.json' 
cfg = load_config(args.config, defaults=RSNA_CFG)
cfg['device'] = args.device
cfg = namedtuple('Struct', cfg.keys())(*cfg.values())
logger.info(cfg)

def takeimg(s):
    return s.split('/')[-1].replace('.jpg', '')

fimg = f'emb/{args.imgemb}.npz'
logger.info(f'Loading : {fimg}')
embname, imgnm = fimg, fimg.replace('.npz', '.imgnames.pk')
imgls = list(map(takeimg, pickle.load( open( imgnm, "rb" ) )))
wtsname = embname.split('/')[-1].replace('.emb.npz', '')
embmat = np.load(embname)['arr_0']
logger.info(f'Weights : {wtsname}')

datadf = pd.read_csv(f'{args.data_dir}/train.csv.zip')
datadf = datadf.set_index('SOPInstanceUID').loc[imgls].reset_index()
folddf = pd.read_csv(f'{args.data_dir}/{args.folds_csv}')

logger.info('Create traindatasets')
trndataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   folddf,
                                   mode="train",
                                   imgclasses=CFG["image_target_cols"],
                                   studyclasses=CFG['exam_target_cols'],
                                   fold=args.fold,
                                   label_smoothing=cfg.label_smoothing,
                                   folds_csv=args.folds_csv)
logger.info('Create valdatasets')
valdataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   folddf,
                                   mode="valid",
                                   imgclasses=CFG["image_target_cols"],
                                   studyclasses=CFG['exam_target_cols'],
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)

logger.info('Create loaders...')
valloader = DataLoader(valdataset, batch_size=cfg.batchsize, shuffle=False, num_workers=4, collate_fn=collateseqfn)
embed_size = embmat.shape[1]
gc.collect()

logger.info('Create model')
if cfg.network == 'transformer':
    model = TransformerNet(cfg)
    output_model_file = f'weights/exam_{cfg.network}_{wtsname}_nlayers{cfg.nlayers}_intermediate{cfg.intermediate_size}_hidden{cfg.hidden_size}_best.bin'

if cfg.network == 'lstm':
    model = LSTMNet(embed_size, 
                       nimgclasses = len(cfg.image_target_cols), 
                       nstudyclasses = len(cfg.exam_target_cols),
                       LSTM_UNITS=cfg.lstm_units, 
                       DO = cfg.dropout)
    output_model_file = f'weights/exam_{cfg.network}_{wtsname}_hidden{cfg.lstm_units}_best.bin'

model = model.to(args.device)
DECAY = 0.0
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = torch.optim.Adam(plist, lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=cfg.lrgamma, last_epoch=-1)


ypredls = []
ypredtstls = []
scaler = torch.cuda.amp.GradScaler()
'''
bce_func_exam = torch.nn.BCEWithLogitsLoss(reduction='none', 
                    weight = torch.tensor(CFG['exam_weights']).to(args.device))
bce_func_img = torch.nn.BCEWithLogitsLoss(reduction='none')
'''
bcewLL_func = torch.nn.BCEWithLogitsLoss(reduction='none')
label_w = torch.tensor(cfg.exam_weights).to(args.device, dtype=torch.float)
image_w = torch.tensor(cfg.image_weight).to(args.device, dtype=torch.float)

def exam_lossfn(studylogits, 
                ystudy, 
                criterion = bcewLL_func, 
                label_w = label_w):
    exam_loss = criterion(studylogits, ystudy)
    exam_wts = exam_loss.shape[0]
    exam_loss = torch.sum(exam_loss*label_w, 1).sum()
    return exam_loss, exam_wts

def image_lossfn(imglogits, 
                 yimg, 
                 mask, 
                 image_w = image_w,
                 criterion = bcewLL_func):
    criterion = bcewLL_func
    qi = yimg.sum(1)/mask.sum(1)
    img_num = mask.sum(1)
    image_loss = (criterion(imglogits.squeeze(-1), yimg) * mask).sum(1)
    image_loss = torch.sum(image_w*qi*image_loss)
    image_wt = torch.sum(image_w*qi*img_num)
    return image_loss, image_wt



logger.info('Start training')
best_val_loss = 100.
for epoch in range(cfg.epochs):
    examsampler = examSampler(trndataset.datadf, trndataset.folddf)
    trnloader = DataLoader(trndataset, batch_size=cfg.batchsize, sampler = examsampler, num_workers=4, collate_fn=collateseqfn)
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    trnres = resultsfn()
    pbartrn = tqdm(enumerate(trnloader), 
                total = len(trndataset)//trnloader.batch_size, 
                desc=f"Train epoch {epoch}", ncols=0)
    for step, batch in pbartrn:
        img_names, yimg, ystudy, masktrn, lelabels = splitbatch(batch, args.device)
        if yimg.sum()==0: 
            logger.info('AAAAAA')
            continue
        xtrn = batch['emb'].to(args.device, dtype=torch.float)
        xtrn = torch.autograd.Variable(xtrn, requires_grad=True)
        yimg = torch.autograd.Variable(yimg)
        ystudy = torch.autograd.Variable(ystudy)
        with autocast():
            
            encoded_layers = model.encoder(xtrn, *model.extended_mask(masktrn))
            imglogits = model.img_linear_out(encoded_layers[-1]).squeeze()
            studylogits = model.study_linear_out(encoded_layers[-1][:, -1]).squeeze()

            exam_loss, exam_wts = exam_lossfn(studylogits, ystudy)
            image_loss, image_wts = image_lossfn(imglogits, yimg, masktrn)
        loss = (exam_loss+image_loss)/(exam_wts+image_wts)
        scaler.scale(loss).backward()
        trnres.loss += (exam_loss+image_loss).item()
        trnres.wts += (exam_wts+image_wts).item()
        trnres.imgloss   += image_loss.item()
        trnres.imgwts    += image_wts.item()
        trnres.exmloss   += exam_loss.item()
        trnres.exmwts    += exam_wts
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        pbartrn.set_postfix({'train loss': trnres.loss/trnres.wts, 
                          'image loss': trnres.imgloss/trnres.imgwts, 
                          'exam loss': trnres.exmloss/trnres.exmwts})
        if step%100==0:
            torch.cuda.empty_cache()  
    
    scheduler.step()
    model.eval()  
    valres = resultsfn()
    pbarval = tqdm(enumerate(valloader), 
                total = len(valdataset)//valloader.batch_size, 
                desc=f"Valid epoch {epoch}", ncols=0)
    for step, batch in pbarval:
        img_names, yimg, ystudy, maskval, lelabels = splitbatch(batch, args.device)
        if yimg.sum()==0: 
            logger.info('AAAAAA')
            continue
        xval = batch['emb'].to(args.device, dtype=torch.float)
        with torch.no_grad():
            
            encoded_layers = model.encoder(xval, *model.extended_mask(maskval))
            imglogits = model.img_linear_out(encoded_layers[-1]).squeeze()
            studylogits = model.study_linear_out(encoded_layers[-1][:, -1]).squeeze()
            
            exam_loss, exam_wts = exam_lossfn(studylogits, ystudy)
            image_loss, image_wts = image_lossfn(imglogits, yimg, maskval)
        loss = (exam_loss+image_loss)/(exam_wts+image_wts)
        valres.loss += (exam_loss+image_loss).item()
        valres.wts += (exam_wts+image_wts).item()
        valres.imgloss   += image_loss.item()
        valres.imgwts    += image_wts.item()
        valres.exmloss   += exam_loss.item()
        valres.exmwts    += exam_wts
        pbarval.set_postfix({'valid loss': valres.loss/valres.wts, 
                          'image loss': valres.imgloss/valres.imgwts, 
                          'exam loss': valres.exmloss/valres.exmwts})
    val_loss = valres.loss/valres.wts
    if best_val_loss>val_loss:
        logger.info(f'Best Epoch {epoch} val loss all {best_val_loss:.4f}')
        best_val_loss = val_loss
        torch.save(model.state_dict(), output_model_file)

