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
PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
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
from training.datasets.classifier_dataset import RSNASequenceDataset, collateseqfn, \
        valSeedSampler, examSampler
from training.zoo.sequence import SpatialDropout, LSTMNet
from training.tools.utils import create_optimizer, AverageMeter, collectPreds, collectLoss
from training.tools.utils import splitbatch, unmasklabels, unmasklogits
from training.losses import getLoss, rsna_criterion_all
from training import losses
from torch.optim.swa_utils import AveragedModel, SWALR
from tensorboardX import SummaryWriter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
logger = get_logger('LSTM', 'INFO') 

import sys; sys.argv=['']; del sys
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
arg('--batchsize', type=int, default=4)
arg('--lr', type=float, default = 0.00001)
arg('--lrgamma', type=float, default = 0.95)
arg('--labeltype', type=str, default='all') # or 'single'
arg('--dropout', type=float, default = 0.2)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--nclasses', type=str, default=1)
arg('--crops-dir', type=str, default='jpegip')
arg('--lstm_units',   type=int, default=512)
arg('--epochs',   type=int, default=12)
arg('--nbags',   type=int, default=12)
arg('--label-smoothing', type=float, default=0.00)
arg('--logdir', type=str, default='logs/b2_1820')
arg("--local_rank", default=0, type=int)
arg('--embrgx', type=str, default='weights/image_weights_regex')
arg("--seed", default=777, type=int)
args = parser.parse_args()

logger.info(f'emb/{args.embrgx}*data.pk')
datals = sorted(glob.glob(f'emb/{args.embrgx}*data.pk'))

def takeimg(s):
    return s.split('/')[-1].replace('.jpg', '')

f=datals[0]
logger.info(f'File load : {f}')
dfname, embname, imgnm = f, f.replace('.data.pk', '.npz'), f.replace('.data.pk', '.imgnames.pk')
datadf = pd.read_pickle(dfname)
embmat = np.load(embname)['arr_0']

imgls = list(map(takeimg, pickle.load( open( imgnm, "rb" ) )))
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
                                   label_smoothing=args.label_smoothing,
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
valloader = DataLoader(valdataset, batch_size=args.batchsize*8, shuffle=False, num_workers=4, collate_fn=collateseqfn)
embed_size = embmat.shape[1]
# del embmat
gc.collect()

logger.info('Create model')
model = LSTMNet(embed_size, 
                       nimgclasses = len(CFG["image_target_cols"]), 
                       nstudyclasses = len(CFG['exam_target_cols']),
                       LSTM_UNITS=args.lstm_units, 
                       DO = args.dropout)
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

ypredls = []
ypredtstls = []
scaler = torch.cuda.amp.GradScaler()
bce_func_exam = torch.nn.BCEWithLogitsLoss(reduction='none', 
                    weight = torch.tensor(CFG['exam_weights']).to(args.device))
bce_func_img = torch.nn.BCEWithLogitsLoss(reduction='none')

logger.info('Start training')
for epoch in range(args.epochs):
    examsampler = examSampler(trndataset.datadf, trndataset.folddf)
    trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = examsampler, num_workers=4, collate_fn=collateseqfn)
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    img_wt = torch.tensor(CFG['image_weight']).to(args.device, dtype=torch.float)
    trncollect = collectPreds()
    valcollect = collectPreds()
    trnloss = collectLoss(trnloader, mode = 'train')
    valloss = collectLoss(valloader, mode = 'valid')
    logger.info(50*'-')
    for step, batch in enumerate(trnloader):
        img_names, yimg, ystudy, masktrn, lelabels = splitbatch(batch, args.device)
        if yimg.sum()==0: 
            logger.info('No positive images in batch')
            continue
        xtrn = batch['emb'].to(args.device, dtype=torch.float)
        xtrn = torch.autograd.Variable(xtrn, requires_grad=True)
        yimg = torch.autograd.Variable(yimg)
        ystudy = torch.autograd.Variable(ystudy)
        with autocast():
            studylogits, imglogits = model(xtrn, masktrn)#.to(args.device, dtype=torch.float)
            yimg, ystudy, lelabels, img_names = unmasklabels(yimg, ystudy, lelabels, img_names, masktrn)
            imglogits, studylogits = unmasklogits(imglogits, studylogits, masktrn)
            # Loss function
            loss, img_loss, exam_loss = rsna_criterion_all(studylogits, 
                                                       ystudy, 
                                                       imglogits, 
                                                       yimg, 
                                                       lelabels, 
                                                       img_wt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        trncollect.append(img_names, lelabels, imglogits, studylogits, yimg, ystudy)
        trnloss.increment(loss, img_loss, exam_loss)
        if step % 50==0: logger.info(f'LOOP:{trnloss.log()}')
        
    output_model_file = f'weights/sequential_lstmepoch{epoch}_fold{args.fold}.bin'
    torch.save(model.state_dict(), output_model_file)
        
    trn_loss, trn_img_loss, trn_exam_loss = rsna_criterion_all(*trncollect.concat(args.device), img_wt)
    logger.info(f'Train loss all {trn_loss:.4f} img {trn_img_loss:.4f} exam {trn_exam_loss:.4f}')
    logger.info(50*'-')
    scheduler.step()
    logger.info('Prep test sub...')
    model.eval()
    for step, batch in enumerate(valloader):
        img_names, yimg, ystudy, maskval, lelabels = splitbatch(batch, args.device)
        xval = batch['emb'].to(args.device, dtype=torch.float)
        studylogits, imglogits = model(xval, maskval)#.to(args.device, dtype=torch.float)
        # Repeat studies to have a prediction for every image
        yimg, ystudy, lelabels, img_names = unmasklabels(yimg, ystudy, lelabels, img_names, maskval)
        imglogits, studylogits = unmasklogits(imglogits, studylogits, maskval)
        loss, img_loss, exam_loss = rsna_criterion_all(studylogits, 
                                                   ystudy, 
                                                   imglogits, 
                                                   yimg, 
                                                   lelabels, 
                                                   img_wt)
        valcollect.append(img_names, lelabels, imglogits, studylogits, yimg, ystudy)
        valloss.increment(loss, img_loss, exam_loss)
    val_loss, val_img_loss, val_exam_loss = rsna_criterion_all(*valcollect.concat(args.device), img_wt)
    logger.info(f'Valid loss all {val_loss:.4f} img {val_img_loss:.4f} exam {val_exam_loss:.4f}')
    
    
    
    
    