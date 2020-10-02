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

'''

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
    #tensor(8613.9717) tensor(2317.2756) tensor(6296.4424)
    #tensor(14386.2285) tensor(7279.) tensor(7107.2480)
    '''
    final_loss = total_loss/total_weights
    return final_loss

# 0.5988
    
# transform into torch tensors
# total_exam_loss, total_img_loss = tensor(2317.2756), tensor(6296.4424)

'''

'''
def groupSum(samples, labels):
    labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    return res

def groupMean(samples, labels):
    labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    res = res / labels_count.float().unsqueeze(1)
    return res

def rsna_criterion(y_pred_exam_, 
                   y_true_exam_, 
                   y_pred_img_, 
                   y_true_img_,
                   le_study, 
                   img_wt, 
                   ):
    logger.info('Exam loss')
    exam_loss = bce_func_exam(y_pred_exam_, y_true_exam_)
    exam_loss = exam_loss.sum(1).unsqueeze(1)
    exam_loss = groupMean(exam_loss, le_study).sum()
    exam_wts = torch.tensor(le_study.unique().shape[0]).float()
    
    logger.info('Image loss')
    image_loss = bce_func_img(y_pred_img_, y_true_img_)
    image_loss = groupSum(image_loss, le_study)
    qi_all = groupMean(y_true_img_, le_study)
    image_loss = (img_wt * qi_all * image_loss).sum()
    img_wts = (img_wt * y_true_img_).sum()
    
    logger.info('Final loss')
    final_loss = (image_loss + exam_loss)/(img_wts + exam_wts)
    return final_loss
'''

import torch
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bce_func_exam = torch.nn.BCELoss(reduction='none', 
                                weight = torch.tensor(CFG['exam_weights']))
bce_func_img = torch.nn.BCELoss(reduction='none')

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
    
    logger.info('Exam loss')
    exam_loss = bce_func_exam(y_pred_exam_, y_true_exam_)
    exam_loss = exam_loss.sum(1).unsqueeze(1)
    exam_loss = groupBy(exam_loss, labels, unique_labels, labels_count, grptype = 'mean').sum()
    exam_wts = torch.tensor(le_study.unique().shape[0]).float()
    
    logger.info('Image loss')
    image_loss = bce_func_img(y_pred_img_, y_true_img_)
    image_loss = groupBy(image_loss, labels, unique_labels, labels_count, grptype = 'sum')
    qi_all = groupBy(y_true_img_, labels, unique_labels, labels_count, grptype = 'mean')
    image_loss = (img_wt * qi_all * image_loss).sum()
    img_wts = (img_wt * y_true_img_).sum()
    
    logger.info('Final loss')
    final_loss = (image_loss + exam_loss)/(img_wts + exam_wts)
    return final_loss

train  = pd.read_csv(f'data/train.csv.zip')
img_label_mean = train[CFG['image_target_cols']].mean(axis=0)
exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
print('===img label mean===\n{} \n\n\n===exam label mean===\n{}\n'.format(img_label_mean, exam_label_mean))
temp_df = train.copy()*0
temp_df[CFG['image_target_cols']] += img_label_mean.values
temp_df[CFG['exam_target_cols']] += exam_label_mean.values
y_true_img = torch.tensor(train[CFG['image_target_cols']].values, dtype=torch.float32)
y_true_exam = torch.tensor(train[CFG['exam_target_cols']].values, dtype=torch.float32) 
y_pred_img = torch.tensor(temp_df[CFG['image_target_cols']].values, dtype=torch.float32)
y_pred_exam = torch.tensor(temp_df[CFG['exam_target_cols']].values, dtype=torch.float32)
# Label encode the study
le_study = torch.tensor(le.fit_transform(train.StudyInstanceUID)).long()
img_wt = torch.tensor(CFG['image_weight'])

%time rsna_criterion(y_pred_exam, \
                   y_true_exam, \
                   y_pred_img, \
                   y_true_img,\
                   le_study, \
                   img_wt)

y_pred_exam_ = y_pred_exam
y_pred_img_  = y_pred_img
y_true_exam_ = y_true_exam
y_true_img_  = y_true_img





'''
def countEncode(labels):
    M = torch.zeros(labels.max()+1, labels.shape[0]) = -1
    M[labels, torch.arange(labels.shape[0])] = labels.float()
    Msum = (M>0).sum(1)[1:]
    grpct = (torch.transpose((M>0)[1:].long(), 0, 1) * Msum).sum(1)
    return grpct
'''


