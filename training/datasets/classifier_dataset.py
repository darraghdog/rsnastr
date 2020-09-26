import os
import random
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
import platform
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils.utils import RSNAWEIGHTS

import albumentations as A
from albumentations.pytorch import ToTensor

from PIL import Image
import cv2
import numpy as np
import pandas as pd
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

class RSNASequenceDataset(Dataset):

    def __init__(self, 
                 datadf,
                 embmat,
                 folddf,
                 mode="train", 
                 fold = 0, 
                 labeltype='all', 
                 label = True,
                 classes=["pe_present_on_image"],
                 label_smoothing=0.01,
                 folds_csv='folds.csv.gz'):
        self.mode = mode
        self.fold = fold
        if mode == "train":
            self.folddf = folddf.query('fold != @self.fold')
        if mode == "valid":
            self.folddf = folddf.query('fold == @self.fold')
        self.classes = classes
        self.label_smoothing = label_smoothing
        self.labeltype = labeltype
        self.datadf = datadf.set_index('StudyInstanceUID')
        self.embmat = embmat
        self.label = label

    def __len__(self):
        return len(self.folddf)

    def __getitem__(self, idx):
        studyidx = self.folddf.iloc[idx].StudyInstanceUID
        studydf = self.datadf.loc[studyidx]
        embidx = (datadf.StudyInstanceUID == studyidx).values
        studyemb = self.embmat[embidx]
        
        imgnames  = studydf.SOPInstanceUID.values
        labels = studydf[self.classes].values
        if self.label:
            if self.mode == 'train': 
                labels = np.clip(labels, self.label_smoothing, 1 - self.label_smoothing)
            
            return {'emb': studyemb, 'img_name' : imgnames, 'labels': labels}
        return {'emb': studyemb, 'img_name' : imgnames}

def collateseqfn(batch):
    maxlen = max([l['emb'].shape[0] for l in batch])
    embdim = batch[0]['emb'].shape[1]
    withlabel = 'labels' in batch[0]
    if withlabel:
        labdim= batch[0]['labels'].shape[1]
        
    for b in batch:
        masklen = maxlen-len(b['emb'])
        b['img_name'] = np.concatenate((np.array(['mask']*masklen), b['img_name']))
        b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withlabel:
            b['labels'] = np.vstack((np.zeros((maxlen-len(b['labels']), labdim)), b['labels']))
            
    outbatch = {'emb' : torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                                for b in batch])).float()}  
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                                for b in batch])).float()
    outbatch['img_name'] =  np.vstack([np.expand_dims(b['img_name'], 0) \
                                                for b in batch])
    if withlabel:
        outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch

class RSNAClassifierDataset(Dataset):

    def __init__(self, 
                 transforms, 
                 mode="train", 
                 fold = 0, 
                 labeltype='all', 
                 imgsize=512,
                 classes=["pe_present_on_image"],
                 crops_dir='jpeg',
                 data_path='data',
                 label_smoothing=0.01,
                 folds_csv='folds.csv.gz'):
        self.mode = mode
        self.fold = fold
        self.fold_csv = folds_csv
        self.crops_dir = crops_dir
        self.classes = classes
        self.datadir = data_path
        self.data = self.loaddf()
        self.transform = transforms
        self.label_smoothing = label_smoothing
        self.imgsize = imgsize
        self.labeltype = labeltype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        try:
        
            samp = self.data.loc[idx]
            study_pe = 0 if samp.negative_exam_for_pe == 1 else 1
            img_name = self.image_file(samp)
            # print(img_name)
            # img_name ='data/jpeg/train/31746ab5e9bc/4308f361d8a4/b96d38eec625.jpg'
            img = self.turboload(img_name)
            if self.imgsize != 512:
                img = cv2.resize(img,(self.imgsize,self.imgsize), interpolation=cv2.INTER_AREA)
            
            if self.transform:       
                augmented = self.transform(image=img)
                img = augmented['image']   
            if self.mode in ['train', 'valid']:
                label = self.data.loc[idx, self.classes]
                if self.mode == 'train': 
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                label = torch.tensor(label)
                if self.labeltype=='all':
                    if label[0] == 0:
                        label[:] = 0 # If image level pe has nothing, then remove the others. 
                return {'img_name': img_name, 'studype': study_pe, 
                        'image': img, 'labels': label}    
            else:      
                return {'img_name': img_name, 'image': img}
        
        except Exception as e:
            print(f'Failed to load {img_name}...{e}')
            return None
        
    def loaddf(self):
        fname = 'train.csv.zip' if self.mode in ['train', 'valid'] else 'test.csv.zip'
        df = pd.read_csv(f'{self.datadir}/{fname}')
        # if we are on Darwin filter
        if platform.system() == 'Darwin':
            self.filter = os.listdir(f'{self.datadir}/{self.crops_dir}/train')
            df = df.query('StudyInstanceUID in @self.filter').reset_index(drop=True)
        fdf = pd.read_csv(f'{self.datadir}/{self.fold_csv}')
        fls = fdf.query('fold == @self.fold').StudyInstanceUID.tolist()
        idx = df.StudyInstanceUID.isin(fls) 
        df = (df[~idx] if self.mode == 'train' else df[idx]).reset_index(drop=True)
        
        return df
    
    # decoding input.jpg to BGR array
    def turboload(self, f):
        in_file = open(f, 'rb')
        bgr_array = jpeg.decode(in_file.read())
        in_file.close()
        return bgr_array[:,:,::-1]
    
    def image_file(self, samp):
        dirtype = 'train' if self.mode != 'test' else 'test'
        return os.path.join(self.datadir, 
                                self.crops_dir,
                                dirtype,
                                samp.StudyInstanceUID,
                                samp.SeriesInstanceUID,
                                samp.SOPInstanceUID) + '.jpg'
    
def collatefn(batch):
    # Remove error reads
    batchout = {\
        'image' : torch.stack([b['image'] for b in batch if b is not None]),
        'labels' : torch.stack([b['labels'] for b in batch if b is not None]),
        'studype' : torch.tensor([b['studype'] for b in batch if b is not None]),
        'img_name' : [b['img_name'] for b in batch if b is not None]}

    return batchout

class nSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data, pe_weight = None, nmin = 2, nmax = 5, seed = None):
        self.data = data
        self.seed = seed
        self.nmin = nmin
        self.nmax = nmax
        self.pe_weight = pe_weight

    def __iter__(self):
        self.sampler = self.sample(self.data)
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)
    
    def sample(self, data):
        # Sample from all studies
        allsamp = self.data.sample(frac= 1, 
                                random_state=self.seed) \
                        .groupby("StudyInstanceUID") \
                        .head(self.nmin).index.tolist()
        totrows = data.iloc[allsamp].shape[0]
        avgrows = data.iloc[allsamp].pe_present_on_image.mean()
        samppos = int(totrows * (1- avgrows) * self.pe_weight)
        # Upsample posotive images, but only take nmax from any single study
        possamp = self.data.query('pe_present_on_image == 1') \
                    .sample(frac= 1, random_state=self.seed) \
                    .groupby("StudyInstanceUID") \
                    .head(self.nmax)[:samppos] \
                    .index.tolist()
        epsamp = list(set(allsamp + possamp ))
        random.shuffle(epsamp)
        
        return epsamp

class valSeedSampler(Sampler):
    r"""Sample N from each of the foloing for validation
    1) Positive samples
    2) Negative samples from positive studies
    3) Images in negative studies
    """

    def __init__(self, data, N = 5000, seed = None):
        self.data = data
        self.seed = seed
        self.N = N
        self.sampler = self.sample(self.data)       

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)
    
    def sample(self, data):
        
        #1) Positive samples
        posimgsamp = self.data.query('negative_exam_for_pe == 0') \
            .query('pe_present_on_image == 1') \
            .sample(frac= 1, random_state=self.seed) \
            .index.tolist()[:self.N]
        #2) Negative samples from positive studies
        negimgsamp = self.data.query('negative_exam_for_pe == 0') \
            .query('pe_present_on_image == 0') \
            .sample(frac= 1, random_state=self.seed) \
            .index.tolist()[:self.N]
        #3) Images in negative studies
        negstdsamp = self.data.query('negative_exam_for_pe == 1') \
            .sample(frac= 1, random_state=self.seed) \
            .index.tolist()[:self.N]
        # Sum them all to one
        epsamp = list(negstdsamp + negimgsamp + posimgsamp )
        
        return epsamp
