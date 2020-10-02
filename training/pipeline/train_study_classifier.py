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
        valSeedSampler
from training.zoo.sequence import SpatialDropout, LSTMNet
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
arg('--label-smoothing', type=float, default=0.00)
arg('--logdir', type=str, default='logs/b2_1820')
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
arg("--embrgx", type=str, default='classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_epoch24__hflip0_transpose0_size320.emb')
args = parser.parse_args()


def takeimg(s):
    return s.split('/')[-1].replace('.jpg', '')

#embrgx = 'classifier_RSNAClassifier_resnext101_32x8d_*__fold*_epoch24__hflip*_transpose0_size320.emb'
#embrgx = 'classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_epoch24__hflip0_transpose0_size320.emb'
datals = sorted(glob.glob(f'emb/{args.embrgx}*data.pk'))
imgls = []
for i, f in enumerate(datals):
    logger.info(f'File load : {f}')
    dfname, embname, imgnm = f, f.replace('.data.pk', '.npz'), f.replace('.data.pk', '.imgnames.pk')
    if i> 1:
        break
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

valmetadf = datadf[datadf.StudyInstanceUID.isin(folddf.query('fold==0').StudyInstanceUID)]
valmetadf = valmetadf.set_index('SOPInstanceUID')
valmeta = valSeedSampler(valmetadf, N = 5000, seed = args.seed)
valmeta = pd.Series([i for i in valmeta])

logger.info('Create loaders...')
trnloader = DataLoader(trndataset, batch_size=args.batchsize, shuffle=True, num_workers=4, collate_fn=collateseqfn)
valloader = DataLoader(valdataset, batch_size=args.batchsize, shuffle=False, num_workers=4, collate_fn=collateseqfn)

'''
batch = []
for i, b in enumerate(trndataset):
    if i>4:break
    batch.append(b)
'''
embed_size = embmat.shape[1]
del embmat
gc.collect()

logger.info('Create model')
model = LSTMNet(embed_size, 
                       nimgclasses = len(CFG["image_target_cols"]), 
                       nstudyclasses = len(CFG['exam_target_cols']),
                       LSTM_UNITS=args.lstm_units, 
                       DO = args.dropout)
model = model.to(args.device)
DECAY = 0.0

# batch = next(iter(trnloader))

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

def groupBy(samples, labels, unique_labels, labels_count, grptype = 'mean'):
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    if grptype == 'sum':
        return res
    if grptype == 'mean':
        res = res / labels_count.float().unsqueeze(1)
        return res

def rsna_criterion(y_pred_exam_, 
                   y_true_exam_, 
                   y_pred_img_, 
                   y_true_img_,
                   le_study, 
                   img_wt, 
                   ):
    # Groupby 
    labels = le_study.view(le_study.size(0), 1).expand(-1, 1)
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    
    #logger.info('Exam loss')
    exam_loss = bce_func_exam(y_pred_exam_, y_true_exam_)
    exam_loss = exam_loss.sum(1).unsqueeze(1)
    exam_loss = groupBy(exam_loss, labels, unique_labels, labels_count, grptype = 'mean').sum()
    exam_wts = torch.tensor(le_study.unique().shape[0]).float()
    
    #logger.info('Image loss')
    image_loss = bce_func_img(y_pred_img_, y_true_img_)
    image_loss = groupBy(image_loss, labels, unique_labels, labels_count, grptype = 'sum')
    qi_all = groupBy(y_true_img_, labels, unique_labels, labels_count, grptype = 'mean')
    image_loss = (img_wt * qi_all * image_loss).sum()
    img_wts = (img_wt * y_true_img_).sum()
    
    #logger.info('Final loss')
    final_loss = (image_loss + exam_loss)/(img_wts + exam_wts)
    return final_loss
    
    
# TODO
#   -- Trouble shoot that sequence is coming out correctly
logger.info('Start training')
for epoch in range(args.epochs):
    #break
    tr_loss = 0.
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    for step, batch in enumerate(trnloader):
        #break
        img_names = batch['img_name']
        yimg = batch['imglabels'].to(args.device, dtype=torch.float)
        ystudy = batch['studylabels'].to(args.device, dtype=torch.float)
        mask = batch['mask'].to(args.device, dtype=torch.int)
        lelabels = batch['lelabels'].to(args.device, dtype=torch.int64)
        img_wt = torch.tensor(CFG['image_weight']).to(args.device, dtype=torch.float)
        x = batch['emb'].to(args.device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        yimg = torch.autograd.Variable(yimg)
        ystudy = torch.autograd.Variable(ystudy)
        with autocast():
            studylogits, imglogits = model(x, mask)#.to(args.device, dtype=torch.float)
            # Repeat studies to have a prediction for every image
            imglogits = imglogits.squeeze()
            studylogits = studylogits.unsqueeze(2).repeat(1, 1, imglogits.size(1))
            ystudy = ystudy.unsqueeze(2).repeat(1, 1, imglogits.size(1))
            studylogits = studylogits.transpose(2, 1)
            ystudy = ystudy.transpose(2, 1)

            # get the mask for masked img labels
            maskidx = mask.view(-1)==1
            # Flatten them all along batch and seq dimension and remove masked values
            yimg = yimg.view(-1, 1)[maskidx]
            imglogits = imglogits.view(-1, 1)[maskidx]
            ystudy = ystudy.reshape(-1, ystudy.size(-1))[maskidx]
            studylogits = studylogits.reshape(-1, ystudy.size(-1))[maskidx]
            lelabels = lelabels.view(-1, 1)[maskidx]
            lelabels = lelabels.flatten()
            
            # Loss function
            loss = rsna_criterion(studylogits, ystudy, imglogits, yimg, lelabels, img_wt)
            
        img_names = img_names.flatten()[maskidx.detach().cpu().numpy()]
        tr_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if (step%100==0) and (step>0):
            logger.info(f'Trn step {step} of {len(trnloader)} trn loss {(tr_loss/(1+step)):.5f}')
    #output_model_file = os.path.join(WORK_DIR, 'weights/lstm_gepoch{}_lstmepoch{}_fold{}.bin'.format(GLOBALEPOCH, epoch, fold))
    output_model_file = f'weights/sequential_lstmepoch{epoch}_fold{args.fold}.bin'
    torch.save(model.state_dict(), output_model_file)
    
    scheduler.step()
    logger.info('Prep test sub...')
    model.eval()
    vallelabels = []
    valimgpreds = []
    valimglabel = []
    valstudylabel = []
    valstudypreds = []
    for step, batch in enumerate(valloader):
        logger.info(step)
        img_names = batch['img_name']
        yimg = batch['imglabels'].to(args.device, dtype=torch.float)
        ystudy = batch['studylabels'].to(args.device, dtype=torch.float)
        mask = batch['mask'].to(args.device, dtype=torch.int)
        lelabels = batch['lelabels'].to(args.device, dtype=torch.int64)
        img_wt = torch.tensor(CFG['image_weight']).to(args.device, dtype=torch.float)
        x = batch['emb'].to(args.device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        yimg = torch.autograd.Variable(yimg)
        ystudy = torch.autograd.Variable(ystudy)
        
        studylogits, imglogits = model(x, mask)#.to(args.device, dtype=torch.float)
        # Repeat studies to have a prediction for every image
        imglogits = imglogits.squeeze()
        studylogits = studylogits.unsqueeze(2).repeat(1, 1, imglogits.size(1))
        ystudy = ystudy.unsqueeze(2).repeat(1, 1, imglogits.size(1))
        studylogits = studylogits.transpose(2, 1)
        ystudy = ystudy.transpose(2, 1)

        # get the mask for masked img labels
        maskidx = mask.view(-1)==1
        # Flatten them all along batch and seq dimension and remove masked values
        yimg = yimg.view(-1, 1)[maskidx]
        imglogits = imglogits.view(-1, 1)[maskidx]
        ystudy = ystudy.reshape(-1, ystudy.size(-1))[maskidx]
        studylogits = studylogits.reshape(-1, ystudy.size(-1))[maskidx]
        lelabels = lelabels.view(-1, 1)[maskidx]
        lelabels = lelabels.flatten().detach().cpu()
        
        # Increment the label encoder
        if len(vallelabels)>0:
            lelabels +=  max(vallelabels[-1]) + 1
        
        vallelabels.append(lelabels)
        valimgpreds.append(imglogits.detach().cpu()  )
        valimglabel.append(yimg.detach().cpu()  )
        valstudylabel.append(ystudy.detach().cpu()  )
        valstudypreds.append(studylogits.detach().cpu()  )
        
    vallelabels = torch.cat(vallelabels).to(args.device)
    valimgpreds = torch.cat(valimgpreds).to(args.device)
    valimglabel = torch.cat(valimglabel).to(args.device)
    valstudylabel = torch.cat(valstudylabel).to(args.device)
    valstudypreds = torch.cat(valstudypreds).to(args.device)

    valloss = rsna_criterion(valstudypreds, valstudylabel, valimgpreds, valimglabel, vallelabels, img_wt)
    logger.info(f'Epoch {epoch} from {args.epochs} val loss {valloss:.5f}')
    del vallelabels, valimgpreds, valimglabel, valstudylabel, valstudypreds
        
    '''
    scheduler.step()
    logger.info('Prep test sub...')
    model.eval()
    valimgls = []
    valpreds = []
    valimglabel = []
    valstudylabel = []
    valstudypreds = []
    for step, batch in enumerate(valloader):
        img_names = batch['img_name']
        ystudy = batch['studylabels'].to(args.device, dtype=torch.float)
        mask = batch['mask'].to(args.device, dtype=torch.int)
        x = batch['emb'].to(args.device, dtype=torch.float)
        yimg = batch['imglabels'].to(args.device, dtype=torch.float)
        ystudy = batch['studylabels'].to(args.device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        studylogits, imglogits = model(x, mask)#.to(args.device, dtype=torch.float)
        # get the mask for masked img labels
        maskidx = mask.view(-1)==1
        imglogits = imglogits.view(-1, 1)[maskidx]
        valpreds += torch.sigmoid(imglogits).detach().cpu().numpy().flatten().tolist()
        valimgls += img_names.flatten()[maskidx.detach().cpu().numpy()].tolist()
        valimglabel += yimg.view(-1, 1)[maskidx].detach().cpu().flatten().tolist()
        # Study level loss
        valstudylabel += ystudy.detach().cpu().numpy().flatten().tolist()
        valstudypreds += torch.sigmoid(studylogits).detach().cpu().numpy().flatten().tolist()
    
    preddf = pd.DataFrame({'pred': valpreds, 'yact': valimglabel }, index = valimgls)
    
    preddf = preddf.loc[valmeta.values]
    yact = datadf.set_index('SOPInstanceUID').loc[preddf.index]
    
    studylloss = log_loss(valstudylabel, valstudypreds)
    logger.info(f'Study level logloss : {studylloss:.5f}')
    
    logger.info('Image level logloss')
    negimg_idx = ((yact.pe_present_on_image < 0.5) & (yact.negative_exam_for_pe < 0.5)).values
    posimg_idx = ((yact.pe_present_on_image > 0.5) & (yact.negative_exam_for_pe < 0.5)).values
    negstd_idx = ((yact.pe_present_on_image < 0.5) & (yact.negative_exam_for_pe > 0.5)).values
    probs = preddf.pred
    targets = preddf.yact
    #logger.info((targets[negimg_idx] == (probs[negimg_idx] > 0.5)))
    #logger.info((targets[negimg_idx] == (probs[negimg_idx] > 0.5)).shape)
    negimg_loss = log_loss(targets[negimg_idx], probs[negimg_idx], labels=[0, 1])
    negimg_acc = (targets[negimg_idx] == (probs[negimg_idx] > 0.5).astype(np.int).values.flatten()).mean()
    posimg_loss = log_loss(targets[posimg_idx], probs[posimg_idx], labels=[0, 1])
    posimg_acc = (targets[posimg_idx] == (probs[posimg_idx] > 0.5).astype(np.int).values.flatten()).mean()
    negstd_loss = log_loss(targets[negstd_idx], probs[negstd_idx], labels=[0, 1])
    negstd_acc = (targets[negstd_idx] == (probs[negstd_idx] > 0.5).astype(np.int).values.flatten()).mean()
    
    avg_acc = (negimg_acc + posimg_acc + negstd_acc) / 3
    avg_loss= (negimg_loss + posimg_loss + negstd_loss) / 3
    log = f'Negimg PosStudy loss {negimg_loss:.4f} acc {negimg_acc:.4f}; '
    log += f'Posimg PosStudy loss {posimg_loss:.4f} acc {posimg_acc:.4f}; '
    log += f'Negimg NegStudy loss {negstd_loss:.4f} acc {negstd_acc:.4f}; '
    log += f'Avg 3 loss {avg_loss:.4f} acc {avg_acc:.4f}'
    logger.info(log)
    '''
    
    
    