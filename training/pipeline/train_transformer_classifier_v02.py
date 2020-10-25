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
try:
    PATH = '/Users/dhanley/Documents/rsnastr'         if platform.system() == 'Darwin' else '/data/rsnastr'
    os.chdir(PATH)
except:
    PATH = '/mount'
    os.chdir(PATH)
sys.path.append(PATH)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss
from utils.logs import get_logger
from utils.utils import RSNAWEIGHTS, RSNA_CFG as CFG
from training.tools.config import load_config
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
from training.datasets.classifier_dataset import RSNASequenceDataset, collateseqfn,         valSeedSampler, examSampler
from training.zoo.sequence import SpatialDropout, LSTMNet, TransformerNet
from training.tools.utils import create_optimizer, AverageMeter, collectPreds, collectLoss
from training.tools.utils import splitbatch, unmasklabels, unmasklogits
from training.losses import getLoss
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


# In[2]:
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
arg('--hidden_size', type=int, default=2048)
arg('--nlayers', type=int, default=1)
arg('--epochs',   type=int, default=12)
arg("--delta", default=False, type=lambda x: (str(x).lower() == 'true'))
arg('--nbags',   type=int, default=12)
arg('--label-smoothing', type=float, default=0.00)
arg('--logdir', type=str, default='logs/b2_1820')
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
arg("--imgembrgx", type=str, default='')
args = parser.parse_args()

class cfg:
    dropout=0.2
    hidden_size=args.hidden_size
    intermediate_size=2048
    max_position_embeddings=1536
    nlayers=args.nlayers
    nheads=8    
    device=args.device
    seed=7

# In[3]:

# In[4]:


def takeimg(s):
    return s.split('/')[-1].replace('.jpg', '')
fimg = sorted(glob.glob(f'emb/{args.imgembrgx}*data.pk'))[0]
logger.info(f'Loading : {fimg}')
dfname, embname, imgnm = fimg, fimg.replace('.data.pk', '.npz'), fimg.replace('.data.pk', '.imgnames.pk')
imgls = list(map(takeimg, pickle.load( open( imgnm, "rb" ) )))
wtsname = embname.split('/')[-1].replace('.emb.npz', '')
embmat = np.load(embname)['arr_0']
logger.info(f'Weights : {wtsname}')

# In[6]:


datadf = pd.read_csv(f'{args.data_dir}/train.csv.zip')
datadf = datadf.set_index('SOPInstanceUID').loc[imgls].reset_index()
folddf = pd.read_csv(f'{args.data_dir}/{args.folds_csv}')


# In[7]:

logger.info('Create traindatasets')
logger.info(f'Embedding delta : {args.delta}')
trndataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   #embexmmat, 
                                   folddf,
                                   mode="train",
                                   delta=args.delta,
                                   imgclasses=CFG["image_target_cols"],
                                   studyclasses=CFG['exam_target_cols'],
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)
logger.info('Create valdatasets')
valdataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   #embexmmat,
                                   folddf,
                                   mode="valid",
                                   delta=args.delta,
                                   imgclasses=CFG["image_target_cols"],
                                   studyclasses=CFG['exam_target_cols'],
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)

def collateseqfn(batch):
    maxlen = cfg.max_position_embeddings
    embdim = batch[0]['emb'].shape[1]
    withimglabel = 'imglabels' in batch[0]
    withstudylabel = 'studylabels' in batch[0]
    if withimglabel:
        labimgdim= batch[0]['imglabels'].shape[1]
        
    for b in batch:
        masklen = maxlen-len(b['emb'])
        b['img_name'] = np.concatenate((np.array(['mask']*masklen), b['img_name']))
        b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withimglabel:
            b['imglabels'] = np.vstack((np.zeros((maxlen-len(b['imglabels']), labimgdim)), b['imglabels']))
            
    outbatch = {'emb' : torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                                for b in batch])).float()}  
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                                for b in batch])).float()
    outbatch['img_name'] = np.vstack([np.expand_dims(b['img_name'], 0) \
                                                for b in batch])
    # Create an label id for each study - label encoder
    outbatch['lelabels'] = torch.ones(outbatch['mask'].shape) * \
                                    torch.arange(len(batch)).unsqueeze(1)

    outbatch['study_name'] = [b['study_name'] for b in batch]
    outbatch['series_name'] = [b['series_name'] for b in batch]
    
    if withimglabel:
        outbatch['imglabels'] = torch.tensor(np.vstack([np.expand_dims(b['imglabels'], 0) for b in batch])).float()
        outbatch['imglabels'] = outbatch['imglabels'].squeeze(-1)
    if withstudylabel:
        outbatch['studylabels'] = torch.tensor(np.concatenate([b['studylabels'] for b in batch], 0))
        
    return outbatch

# In[10]:


logger.info('Create loaders...')
valloader = DataLoader(valdataset, batch_size=args.batchsize, shuffle=False, num_workers=4, collate_fn=collateseqfn)
embed_size = embmat.shape[1]
if args.delta:
    embed_size = embed_size * 3
gc.collect()
'''
chkloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collateseqfn)
chkbatch = next(iter(chkloader))
logger.info('Weights')
logger.info(embname)
logger.info('Embeddings')
logger.info(chkbatch['emb'])
logger.info('Mask')
logger.info(chkbatch['mask'])
logger.info('Img name')
logger.info(chkbatch['img_name'])
logger.info('Study Name')
logger.info(chkbatch['study_name'])
'''
# In[11]:

logger.info('Create model')
'''
model = LSTMNet(embed_size, 
                       nimgclasses = len(CFG["image_target_cols"]), 
                       nstudyclasses = len(CFG['exam_target_cols']),
                       LSTM_UNITS=args.lstm_units, 
                       DO = args.dropout)
'''
model = TransformerNet(cfg)
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


# In[12]:


ypredls = []
ypredtstls = []
scaler = torch.cuda.amp.GradScaler()
bce_func_exam = torch.nn.BCEWithLogitsLoss(reduction='none', 
                    weight = torch.tensor(CFG['exam_weights']).to(args.device))
bce_func_img = torch.nn.BCEWithLogitsLoss(reduction='none')


# In[13]:


bcewLL_func = torch.nn.BCEWithLogitsLoss(reduction='none')
label_w = torch.tensor([0.0736196319, 
             0.2346625767, 
             0.0782208589, 
             0.06257668712, 
             0.1042944785, 
             0.06257668712, 
             0.1042944785, 
             0.1877300613, 
             0.09202453988]).to(args.device, dtype=torch.float)
image_w = torch.tensor(0.07361963).to(args.device, dtype=torch.float)

def exam_lossfn(studylogits, ystudy, criterion = bcewLL_func):
    exam_loss = criterion(studylogits, ystudy)
    exam_wts = exam_loss.shape[0]
    exam_loss = torch.sum(exam_loss*label_w, 1).sum()
    return exam_loss, exam_wts

def image_lossfn(imglogits, yimg, mask, criterion = bcewLL_func):
    criterion = bcewLL_func
    qi = yimg.sum(1)/mask.sum(1)
    img_num = mask.sum(1)
    image_loss = (criterion(imglogits.squeeze(-1), yimg) * mask).sum(1)
    image_loss = torch.sum(image_w*qi*image_loss)
    image_wt = torch.sum(image_w*qi*img_num)
    return image_loss, image_wt

class resultsfn:
    loss   = 0.
    wts    = 0.
    imgloss   = 0.
    imgwts    = 0.
    exmloss   = 0.
    exmwts    = 0.

logger.info('Start training')
best_val_loss = 100.
for epoch in range(args.epochs):
    examsampler = examSampler(trndataset.datadf, trndataset.folddf)
    trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = examsampler, num_workers=4, collate_fn=collateseqfn)
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

    #logger.info(f'Epoch {epoch} train loss all {trnres.loss/trnres.wts:.4f}')
    deltamsg = '_delta' if args.delta else ''
    output_model_file = f'weights/exam_transformer_{wtsname}{deltamsg}_nlayers{args.nlayers}_hidden{args.hidden_size}__epoch{epoch}.bin'
    torch.save(model.state_dict(), output_model_file)
    
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

