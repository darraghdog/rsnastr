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

from torch.utils.data import Dataset
class RSNASequenceDataset(Dataset):

    def __init__(self, 
                 datadf,
                 embimgmat,
                 # embexmmat,
                 folddf,
                 mode="train", 
                 delta=False,
                 fold = 0, 
                 labeltype='all', 
                 label = True,
                 imgclasses=["pe_present_on_image"],
                 studyclasses=["pe_present_on_image"],
                 label_smoothing=0.01,
                 folds_csv='folds.csv.gz'):
        self.mode = mode
        self.fold = fold        
        self.datadf = datadf.set_index('StudyInstanceUID')
        if mode == "train":
            self.folddf = folddf.query('fold != @self.fold')
            idx = self.folddf.StudyInstanceUID.isin(self.datadf.index)
            self.folddf = self.folddf[idx].reset_index(drop=True)
        if mode == "all":
            1 # Keep all - for embeddings
        if mode == "valid":
            self.folddf = folddf.query('fold == @self.fold')
        self.folddf = pd.merge(self.folddf, self.datadf.reset_index() \
                          .filter(regex='Study|Series').drop_duplicates())
        self.imgclasses = imgclasses
        self.studyclasses = studyclasses
        self.label_smoothing = label_smoothing
        self.labeltype = labeltype
        self.embimgmat = embimgmat
        #self.embexmmat = embexmmat
        self.label = label
        self.delta = delta

    def __len__(self):
        return len(self.folddf)

    def __getitem__(self, idx):
        # idx = 0
        studyidx = self.folddf.iloc[idx].StudyInstanceUID
        seriesidx = self.folddf.iloc[idx].SeriesInstanceUID
        studydf = self.datadf.loc[studyidx].query('SeriesInstanceUID == @seriesidx')
        embidx = self.datadf.index == studyidx

        #studyimgemb = self.embimgmat[embidx]
        #studyexmemb = self.embexmmat[embidx]
        #studyemb = np.concatenate((self.embimgmat[embidx],
        #                           self.embexmmat[embidx]),1)
        studyemb = self.embimgmat[embidx]
        
        if self.delta:
            studydeltalag  = np.zeros(studyemb.shape)
            studydeltalead = np.zeros(studyemb.shape)
            studydeltalag [1:] = studyemb[1:]-studyemb[:-1]
            studydeltalead[:-1] = studyemb[:-1]-studyemb[1:]
            studyemb = np.concatenate((studyemb, studydeltalag, studydeltalead), -1)
        
        imgnames  = studydf.SOPInstanceUID.values
        
        out = {'emb': studyemb, 
               'img_name' : imgnames, 
               'study_name' : studyidx, 
               'series_name' : seriesidx}
        
        if self.mode == 'test':
            return out
        
        if 'pe_present_on_image' in self.imgclasses :
            out['imglabels'] = studydf[self.imgclasses].values
            
        if len(self.studyclasses) > 0:
            out['studylabels'] = studydf[self.studyclasses].drop_duplicates().values

        if self.mode == 'train': 
                out['studylabels'] = np.clip(out['studylabels'], self.label_smoothing, 1 - self.label_smoothing)
                out['imglabels'] = np.clip(out['imglabels'], self.label_smoothing, 1 - self.label_smoothing)
        return out


def collateseqfn(batch):
    maxlen = max([l['emb'].shape[0] for l in batch])
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
    
    
class RSNAImageSequenceDataset(Dataset):

    def __init__(self, 
                 transforms, 
                 pos_sample_weight = 1.,
                 sample_count = 12, 
                 imgsize=512,
                 mode="train", 
                 data_path = 'data',
                 crops_dir='jpegip',
                 fold = 0, 
                 labeltype='all', 
                 balanced=True, 
                 label = True,
                 imgclasses=["pe_present_on_image"],
                 studyclasses=["pe_present_on_image"],
                 label_smoothing=0.01,
                 folds_csv='folds.csv.gz'):
        self.mode = mode
        self.imgclasses = imgclasses
        self.studyclasses = studyclasses
        self.crops_dir = crops_dir
        self.datadir = data_path
        self.label_smoothing = label_smoothing
        self.labeltype = labeltype
        self.label = label
        self.sample_count = sample_count
        self.balanced = balanced
        self.datadir = data_path
        self.crops_dir = crops_dir
        self.fold = fold        
        self.imgsize = imgsize
        self.transform = transforms
        self.folds_csv = folds_csv
        self.datadf = self.loaddf()
        self.folddf = pd.read_csv(f'{self.datadir}/{self.folds_csv}')
        self.folddf = pd.merge(self.folddf, self.datadf.reset_index() \
                          .filter(regex='Study|Series|negative_exam').drop_duplicates())
        self.pos_sample_weight = pos_sample_weight

    def __len__(self):
        return len(self.folddf)

    def __getitem__(self, idx):
        # idx = 1
        studyidx = self.folddf.iloc[idx].StudyInstanceUID
        seriesidx = self.folddf.iloc[idx].SeriesInstanceUID
        studydf = self.datadf.query('StudyInstanceUID == @studyidx')\
                        .query('SeriesInstanceUID == @seriesidx')
        
        # Evenly weight pos and negative
        if self.balanced and (studydf["pe_present_on_image"].values.sum() > 0): 
            selection_weight = float(self.pos_sample_weight) * \
                            studydf["pe_present_on_image"].values / \
                                (studydf["pe_present_on_image"].values.sum())
            selection_weight[selection_weight==0] = 1 / \
                            (studydf["pe_present_on_image"]==0).sum()
        else:
            selection_weight = np.ones(len(studydf))
            
        sample_count = self.sample_count \
            if studydf.shape[0] >= self.sample_count else studydf.shape[0]
        samp = studydf.sample(sample_count, 
                   weights=selection_weight).SOPInstanceUID.values

        studydf = studydf[studydf.SOPInstanceUID.isin(samp)]
        
        try:
            imgs = []
            for i, samp in studydf.reset_index().iterrows():
                try:
                    img_name = self.image_file(samp)
                    # print(img_name)
                    # img_name ='data/jpeg/train/31746ab5e9bc/4308f361d8a4/b96d38eec625.jpg'
                    img = self.turboload(img_name)
                    if self.imgsize != 512:
                        img = cv2.resize(img,(self.imgsize,self.imgsize), interpolation=cv2.INTER_AREA)
                    if self.transform:       
                        augmented = self.transform(image=img)
                        img = augmented['image']   
                    imgs.append(img)
                except Exception as e:
                    imgs.append(img)
                    print(f'Failed to load {img_name}...{e}')
            imgs = torch.stack(imgs)
            if self.mode in ['train', 'valid', 'all']:
                label = studydf[self.imgclasses + self.studyclasses].values
                if self.mode == 'train': 
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                label = torch.tensor(label).float()
                return {'studype': studyidx, 
                        'image': imgs, 'labels': label}    
            else:      
                return {'img_name': img_name, 'image': imgs}
        
        except Exception as e:
            print(f'Failed to load {img_name}...{e}')
            return None
        
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
    def loaddf(self):
        fname = 'train.csv.zip' if self.mode in ['train', 'valid', 'all'] else 'test.csv.zip'
        df = pd.read_csv(f'{self.datadir}/{fname}')
        # if we are on Darwin filter
        if platform.system() == 'Darwin':
            self.filter = os.listdir(f'{self.datadir}/{self.crops_dir}/train')
            df = df.query('StudyInstanceUID in @self.filter').reset_index(drop=True)
        fdf = pd.read_csv(f'{self.datadir}/{self.folds_csv}')
        fls = fdf.query('fold == @self.fold').StudyInstanceUID.tolist()
        idx = df.StudyInstanceUID.isin(fls) 
        if self.mode == 'train':
            df = (df[~idx]).reset_index(drop=True)
        if self.mode == 'valid':
            df = (df[idx]).reset_index(drop=True)
        return df
    
def collateseqimgfn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]

    # Pad with zero frames
    x_batch = torch.stack([b['image'] for b in batch])
    nm_batch = np.array([b['studype'] for b in batch])
    
    if 'labels' in batch[0]:
        y_batch = torch.stack([b['labels'] for b in batch])
        return {'image': x_batch, 'study': nm_batch, 'labels': y_batch}
    else:
        return {'image': x_batch, 'study': nm_batch}



class RSNAClassifierDataset(Dataset):

    def __init__(self, 
                 transforms, 
                 mode="train", 
                 fold = 0, 
                 labeltype='all', 
                 imgsize=512,
                 imgclasses=["pe_present_on_image"],
                 studyclasses=["pe_present_on_image"],
                 crops_dir='jpeg',
                 data_path='data',
                 label_smoothing=0.01,
                 folds_csv='folds.csv.gz'):
        self.mode = mode
        self.fold = fold
        self.fold_csv = folds_csv
        self.crops_dir = crops_dir
        self.imgclasses = imgclasses
        self.studyclasses = studyclasses
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
            if self.mode in ['train', 'valid', 'all']:
                label = self.data.loc[idx, self.imgclasses + self.studyclasses]
                if self.mode == 'train': 
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                label = torch.tensor(label)
                if self.labeltype=='all':
                    if label[:2].sum() < 0.1:
                        label[:] = 0 # If image level pe has nothing, then remove the others. 
                return {'img_name': img_name, 'studype': study_pe, 
                        'image': img, 'labels': label}
            else:      
                return {'img_name': img_name, 'image': img}
        
        except Exception as e:
            print(f'Failed to load {img_name}...{e}')
            return None
        
    def loaddf(self):
        fname = 'train.csv.zip' if self.mode in ['train', 'valid', 'all'] else 'test.csv.zip'
        df = pd.read_csv(f'{self.datadir}/{fname}')
        # if we are on Darwin filter
        if platform.system() == 'Darwin':
            self.filter = os.listdir(f'{self.datadir}/{self.crops_dir}/train')
            df = df.query('StudyInstanceUID in @self.filter').reset_index(drop=True)
        fdf = pd.read_csv(f'{self.datadir}/{self.fold_csv}')
        fls = fdf.query('fold == @self.fold').StudyInstanceUID.tolist()
        idx = df.StudyInstanceUID.isin(fls) 
        if self.mode == 'train':
            df = (df[~idx]).reset_index(drop=True)
        if self.mode == 'valid':
            df = (df[idx]).reset_index(drop=True)
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

    def __init__(self, data, 
                 pe_weight = None, 
                 nmin = 2, 
                 nmax = 5, 
                 examlevel = False,
                 seed = None):
        self.data = data
        self.seed = seed
        self.nmin = nmin
        self.nmax = nmax
        self.pe_weight = pe_weight
        self.examlevel = examlevel

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

    def __init__(self, 
                 data, 
                 N = 5000, 
                 examlevel = False, 
                 seed = None):
        self.data = data
        self.seed = seed
        self.N = N
        self.examlevel = examlevel
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
        if self.examlevel:
            epsamp = list(negstdsamp + posimgsamp )
        else:
            epsamp = list(negstdsamp + negimgsamp + posimgsamp )
        
        return epsamp


class examSampler(Sampler):
    r"""Sample N from each of the foloing for validation
    1) Positive samples
    2) Negative samples from positive studies
    3) Images in negative studies
    """

    def __init__(self, data, folddf, seed = None):
        self.data = data.copy()
        self.folddf = folddf.copy()
        self.seed = seed
        self.sampler = self.sample(self.data, self.folddf)       

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)
    
    def sample(self, data, folddf):
        
        folddf = folddf.reset_index(drop=True)
        alldf = data['negative_exam_for_pe'].loc[folddf.StudyInstanceUID] \
                .reset_index().drop_duplicates().reset_index(drop=True)
        posdf = alldf.query('negative_exam_for_pe==1').sample(frac= 1)
        negdf = alldf.query('negative_exam_for_pe==0').sample(frac= 1)
        posdf['seq'] = np.arange(posdf.shape[0])/posdf.shape[0]
        negdf['seq'] = np.arange(negdf.shape[0])/negdf.shape[0]
        StudyInstanceUIDSeq = pd.concat([posdf, negdf], 0).sort_values('seq').StudyInstanceUID
        sample_idx = folddf.reset_index().set_index('StudyInstanceUID').loc[StudyInstanceUIDSeq]['index'].values
        
        return sample_idx
