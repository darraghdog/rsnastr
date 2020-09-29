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
from utils.utils import RSNAWEIGHTS
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
from training.datasets.classifier_dataset import RSNASequenceDataset, collateseqfn
from training.zoo.sequence import SpatialDropout
from training.tools.utils import create_optimizer, AverageMeter
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
arg('--label-smoothing', type=float, default=0.01)
arg('--logdir', type=str, default='logs/b2_1820')
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
args = parser.parse_args()

def takeimg(s):
    return s.split('/')[-1].replace('.jpg', '')

embrgx = 'classifier_RSNAClassifier_resnext101_32x8d_*__fold*_epoch24__hflip*_transpose0_size320.emb'
embrgx = 'classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_epoch24__hflip0_transpose0_size320.emb'
datals = sorted(glob.glob(f'emb/{embrgx}*data.pk'))
imgls = []
for i, f in enumerate(datals):
    logger.info(f'File load : {f}')
    dfname, embname, imgnm = f, f.replace('.data.pk', '.npz'), f.replace('.data.pk', '.imgnames.pk')
    if i == 0:
        datadf = pd.read_pickle(dfname)
        embmat = np.load(embname)['arr_0']
    if i>0:
        embmat = np.append( embmat, np.load(embname)['arr_0'], 0)
        datadf = pd.concat([datadf, pd.read_pickle(dfname)], 0)
    imgls += list(map(takeimg, pickle.load( open( imgnm, "rb" ) )))
    logger.info(f'Embedding shape : {embmat.shape}')
    logger.info(f'DataFrame shape : {datadf.shape}')
    logger.info(f'DataFrame shape : {len(imgls)}')
    gc.collect()
folddf = pd.read_csv(f'{args.data_dir}/{args.folds_csv}')
datadf = datadf.set_index('SOPInstanceUID').loc[imgls].reset_index()
datadf.iloc[0]
datadf.pe_present_on_image[:10000].plot()

'''
batch = []
for b in trndataset:
    batch.append(b)
    if len(batch)>7:
        break
'''

logger.info('Create traindatasets')
trndataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   folddf,
                                   mode="train",
                                   classes=["pe_present_on_image", "negative_exam_for_pe"],
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)
logger.info('Create valdatasets')
valdataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   folddf,
                                   mode="valid",
                                   classes=["pe_present_on_image", "negative_exam_for_pe"],
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)

logger.info('Create loaders...')
trnloader = DataLoader(trndataset, batch_size=args.batchsize, shuffle=True, num_workers=4, collate_fn=collateseqfn)
valloader = DataLoader(valdataset, batch_size=args.batchsize, shuffle=False, num_workers=4, collate_fn=collateseqfn)
embed_size = embmat.shape[1]
del embmat
gc.collect()


# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet(nn.Module):
    def __init__(self, embed_size, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        
        self.embed_size = embed_size
        self.embedding_dropout = SpatialDropout(0.0) #DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.img_linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.img_linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.study_linear1 = nn.Linear(LSTM_UNITS*4, LSTM_UNITS*4)

        self.img_linear_out = nn.Linear(LSTM_UNITS*2, 1)
        self.study_linear_out = nn.Linear(LSTM_UNITS*4, 1)

    def forward(self, x, mask, lengths=None):
        
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # Masked mean and max pool for study level prediction
        avg_pool = torch.sum(h_lstm2, 1) * (1/ mask.sum(1)).unsqueeze(1)
        max_pool, _ = torch.max(h_lstm2, 1)
        
        # Get study level prediction
        h_study_conc = torch.cat((max_pool, avg_pool), 1)
        h_study_conc_linear1  = nn.functional.relu(self.study_linear1(h_study_conc))
        study_hidden = h_study_conc + h_study_conc_linear1
        study_output = self.study_linear_out(study_hidden)
        
        # Get study level prediction
        h_img_conc_linear1  = nn.functional.relu(self.img_linear1(h_lstm1))
        h_img_conc_linear2  = nn.functional.relu(self.img_linear2(h_lstm2))
        img_hidden = h_lstm1 + h_lstm2 + h_img_conc_linear1 + h_img_conc_linear2 # + h_embadd
        img_output = self.img_linear_out(img_hidden)
        
        return study_output, img_output

logger.info('Create model')
model = NeuralNet(embed_size, LSTM_UNITS=args.lstm_units, DO = args.dropout)
model = model.to(args.device)
DECAY = 0.0

batch = next(iter(trnloader))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = torch.optim.Adam(plist, lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lrgamma, last_epoch=-1)
criterion = torch.nn.BCEWithLogitsLoss()

ypredls = []
ypredtstls = []
if args.device != 'cpu':
    scaler = torch.cuda.amp.GradScaler()
    
    
# TODO
#   -- Add on study names to the batch
#   -- Trouble shoot that sequence is coming out correctly

for epoch in range(args.epochs):
    tr_loss = 0.
    for param in model.parameters():
        param.requires_grad = True
    model.train()  

    for step, batch in enumerate(trnloader):
        img_names = batch['img_name']
        yimg = batch['imglabels'].to(args.device, dtype=torch.float)
        ystudy = batch['studylabels'].to(args.device, dtype=torch.float)
        mask = batch['mask'].to(args.device, dtype=torch.int)
        x = batch['emb'].to(args.device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        yimg = torch.autograd.Variable(yimg)
        ystudy = torch.autograd.Variable(ystudy)
        with autocast():
            studylogits, imglogits = model(x, mask)#.to(args.device, dtype=torch.float)
            # get the mask for masked img labels
            maskidx = mask.view(-1)==1
            yimg = yimg.view(-1, 1)[maskidx]
            img_names = img_names.flatten()[maskidx]
            imglogits = imglogits.view(-1, 1)[maskidx]
            # Get img loss
            loss1 = criterion(imglogits, yimg)
            # Get the study loss
            loss2 = criterion(studylogits, ystudy)
            # Try average of both losses for the start
            loss = 0.5 * (loss1 + loss2)

        tr_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step%1000==0:
            logger.info('Trn step {} of {} trn lossavg {:.5f}'. \
                        format(step, len(trnloader), (tr_loss/(1+step))))
    output_model_file = os.path.join(WORK_DIR, 'weights/lstm_gepoch{}_lstmepoch{}_fold{}.bin'.format(GLOBALEPOCH, epoch, fold))
    torch.save(model.state_dict(), output_model_file)

    scheduler.step()
    logger.info('Prep test sub...')
    model.eval()
    valimgls = []
    valpreds = []
    for step, batch in enumerate(valloader):
        img_names = batch['img_name']
        ystudy = batch['studylabels'].to(args.device, dtype=torch.float)
        mask = batch['mask'].to(args.device, dtype=torch.int)
        x = batch['emb'].to(args.device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        studylogits, imglogits = model(x, mask)#.to(args.device, dtype=torch.float)
        # get the mask for masked img labels
        maskidx = mask.view(-1)==1
        imglogits = imglogits.view(-1, 1)[maskidx]
        valpreds = torch.sigmoid(imglogits).detach().cpu().numpy().flatten().tolist()
        valimgls = img_names.flatten()[maskidx]
    
    preddf = pd.DataFrame({'pred': valpreds}, index = valimgls)
    
logger.info('Write out bagged prediction to preds folder')
ytstpred = sum(ypredtstls[-nbags:])/len(ypredtstls[-nbags:])
ytstout = makeSub(ytstpred, imgtst)
ytstout.to_csv('preds/lstm{}{}{}_{}_epoch{}_sub_{}.csv.gz'.format(TTAHFLIP, TTATRANSPOSE, LSTM_UNITS, WORK_DIR.split('/')[-1], epoch, embnm), \
            index = False, compression = 'gzip')
