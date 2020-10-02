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
    if i>1: continue
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
                                weight = torch.tensor(CFG['exam_weights']))
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
    break
    tr_loss = 0.
    tr_loss1 = 0.
    tr_loss2 = 0.
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    #break
    for step, batch in enumerate(trnloader):
        break
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
        tr_loss1 += loss1.item()
        tr_loss2 += loss2.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step%250==0:
            logger.info('Trn step {} of {} trn lossavg {:.5f} img loss {:.5f} study loss {:.5f}'. \
                        format(step, len(trnloader), (tr_loss/(1+step)), (tr_loss1/(1+step)), (tr_loss2/(1+step))))
    #output_model_file = os.path.join(WORK_DIR, 'weights/lstm_gepoch{}_lstmepoch{}_fold{}.bin'.format(GLOBALEPOCH, epoch, fold))
    output_model_file = f'weights/sequential_lstmepoch{epoch}_fold{args.fold}.bin'
    torch.save(model.state_dict(), output_model_file)

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
    probdf = pd.DataFrame({'img': img_names, 
                           'label': targets.values.flatten(),
                           'studype': targets.values.flatten(),
                           'probs': probs.values.flatten()})
    '''


CFG = {
    'image_target_cols': [
        'pe_present_on_image', # only image level
    ],
    'exam_target_cols': [
        'negative_exam_for_pe', # exam level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ], 
    'image_weight': 0.07361963,
    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988],
}

train  = pd.read_csv(f'data/train.csv.zip')
img_label_mean = train[CFG['image_target_cols']].mean(axis=0)
exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
print('===img label mean===\n{} \n\n\n===exam label mean===\n{}\n'.format(img_label_mean, exam_label_mean))
temp_df = train.copy()*0
temp_df[CFG['image_target_cols']] += img_label_mean.values
temp_df[CFG['exam_target_cols']] += exam_label_mean.values

loss = rsna_torch_wloss(CFG, train[CFG['image_target_cols']].values, train[CFG['exam_target_cols']].values, 
                      temp_df[CFG['image_target_cols']].values, temp_df[CFG['exam_target_cols']].values, 
                      list(train.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))



# transform into torch tensors
y_true_img = torch.tensor(train[CFG['image_target_cols']].values, dtype=torch.float32)
y_true_exam = torch.tensor(train[CFG['exam_target_cols']].values, dtype=torch.float32) 
y_pred_img = torch.tensor(temp_df[CFG['image_target_cols']].values, dtype=torch.float32)
y_pred_exam = torch.tensor(temp_df[CFG['exam_target_cols']].values, dtype=torch.float32)
chunk_sizes = list(train.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count())

bce_func_exam = torch.nn.BCELoss(reduction='sum', 
                                weight = torch.tensor(CFG['exam_weights']))

def rsna_torch_wloss(CFG, y_true_img, y_true_exam, y_pred_img, y_pred_exam, chunk_sizes):
    
    # split into chunks (each chunks is for a single exam)
    y_true_img_chunks = torch.split(y_true_img, chunk_sizes, dim=0) 
    y_true_exam_chunks = torch.split(y_true_exam, chunk_sizes, dim=0) 
    y_pred_img_chunks = torch.split(y_pred_img, chunk_sizes, dim=0)
    y_pred_exam_chunks = torch.split(y_pred_exam, chunk_sizes, dim=0)
    
    label_w = torch.tensor(CFG['exam_weights']).view(1, -1)
    img_w = CFG['image_weight']
    bce_func = torch.nn.BCELoss(reduction='none')
    
    total_img_weights = torch.tensor(0, dtype=torch.float32)
    total_exam_weights = torch.tensor(0, dtype=torch.float32)
    total_img_loss = torch.tensor(0, dtype=torch.float32)
    total_exam_loss= torch.tensor(0, dtype=torch.float32)
    total_loss = torch.tensor(0, dtype=torch.float32)
    total_weights = torch.tensor(0, dtype=torch.float32)
    
    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in \
            enumerate(zip(y_true_img_chunks, 
                          y_true_exam_chunks, 
                          y_pred_img_chunks, 
                          y_pred_exam_chunks)):
        exam_loss = bce_func_exam(y_pred_exam_[0, :], y_true_exam_[0, :])
        
        image_loss = bce_func(y_pred_img_, y_true_img_)
        img_num = chunk_sizes[i]
        qi = torch.sum(y_true_img_)/img_num
        image_loss = torch.sum(img_w*qi*image_loss)
        
        total_exam_loss += exam_loss
        total_img_loss += image_loss
        total_loss += exam_loss+image_loss
        total_exam_weights += label_w.sum() 
        total_img_weights += img_w*qi*img_num
        total_weights += label_w.sum() + img_w*qi*img_num
        #print(exam_loss, image_loss, img_num);assert False
    print(total_loss, total_exam_loss, total_img_loss)
    print(total_weights, total_exam_weights, total_img_weights)
    '''
    tensor(8613.9717) tensor(2317.2756) tensor(6296.4424)
    tensor(14386.2285) tensor(7279.) tensor(7107.2480)
    '''
    final_loss = total_loss/total_weights
    return final_loss

# 0.5988
    
# transform into torch tensors
# total_exam_loss, total_img_loss = tensor(2317.2756), tensor(6296.4424)



import torch
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bce_func_exam = torch.nn.BCELoss(reduction='none', 
                                weight = torch.tensor(CFG['exam_weights']))
bce_func_img = torch.nn.BCELoss(reduction='none')

train  = pd.read_csv(f'data/train.csv.zip')
img_label_mean = train[CFG['image_target_cols']].mean(axis=0)
exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
print('===img label mean===\n{} \n\n\n===exam label mean===\n{}\n'.format(img_label_mean, exam_label_mean))
temp_df = train.copy()*0
temp_df[CFG['image_target_cols']] += img_label_mean.values
temp_df[CFG['exam_target_cols']] += exam_label_mean.values
y_studies = torch.tensor(le.fit_transform(train.StudyInstanceUID)).long()
y_true_img = torch.tensor(train[CFG['image_target_cols']].values, dtype=torch.float32)
y_true_exam = torch.tensor(train[CFG['exam_target_cols']].values, dtype=torch.float32) 
y_pred_img = torch.tensor(temp_df[CFG['image_target_cols']].values, dtype=torch.float32)
y_pred_exam = torch.tensor(temp_df[CFG['exam_target_cols']].values, dtype=torch.float32)

# Exam loss
label_w = torch.tensor(CFG['exam_weights']).view(1, -1)
exam_loss = bce_func_exam(y_pred_exam, y_true_exam)
exam_loss = exam_loss.sum(1).unsqueeze(1)
exam_loss = groupMean(exam_loss, y_studies).sum()
exam_wts = label_w .sum() * y_studies.unique().shape[0]

# Image loss
img_w = CFG['image_weight']
image_loss = bce_func_img(y_pred_img, y_true_img)
image_loss = groupSum(image_loss, y_studies)
qi_all = groupMean(y_true_img, y_studies)
image_loss = (img_w * qi_all * image_loss).sum()
img_wts = (img_w * y_true_img).sum()

# Final loss
final_loss = (image_loss + exam_loss)/(img_wts + exam_wts)



def groupSum(samples, labels):
    '''
    https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/2?u=darragh_hanley
    '''
    labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    return res

def groupMean(samples, labels):
    '''
    https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/2?u=darragh_hanley
    '''
    labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    res = res / labels_count.float().unsqueeze(1)
    return res

def countEncode(labels):
    '''
    # The label must not have a zero
    labels1 = torch.LongTensor([1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 6, 6])
    countEncode(labels1)
    -->> tensor([4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 1])
    '''
    M = torch.zeros(labels.max()+1, labels.shape[0]) = -1
    M[labels, torch.arange(labels.shape[0])] = labels.float()
    Msum = (M>0).sum(1)[1:]
    grpct = (torch.transpose((M>0)[1:].long(), 0, 1) * Msum).sum(1)
    return grpct



chunk_sizes = list(train.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count())



with torch.no_grad():
    loss = rsna_torch_wloss(CFG, train[CFG['image_target_cols']].values, train[CFG['exam_target_cols']].values, 
                      temp_df[CFG['image_target_cols']].values, temp_df[CFG['exam_target_cols']].values, 
                      list(train.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))

    print(loss)
    
    
samples = torch.Tensor([
                     [0.1, 0.1],    #-> group / class 1
                     [0.2, 0.2],    #-> group / class 1
                     [0.2, 0.2],    #-> group / class 2
                     [0.2, 0.2],    #-> group / class 2
                     [0.4, 0.4],    #-> group / class 2
                     [0.6, 0.6]     #-> group / class 0
              ])
labels = torch.LongTensor([1, 1, 2, 2, 2, 0])

labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
res = res / labels_count.float().unsqueeze(1)


def countEncode(labels):
    '''
    # The label must not have a zero
    labels1 = torch.LongTensor([1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 6, 6])
    countEncode(labels1)
    -->> tensor([4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 1])
    '''
    M = torch.zeros(labels.max()+1, labels.shape[0]) = -1
    M[labels, torch.arange(labels.shape[0])] = labels.float()
    Msum = (M>0).sum(1)[1:]
    grpct = (torch.transpose((M>0)[1:].long(), 0, 1) * Msum).sum(1)
    return grpct








M[labels, torch.arange(samples.shape[0])] = samples[:,0]
M.sum(1)
M.sum(1)
M = torch.nn.functional.normalize(M, p=1, dim=1)
torch.mm(M, samples)