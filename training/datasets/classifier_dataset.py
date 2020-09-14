import os
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
import pandas as pd
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()



class RSNAClassifierDataset(Dataset):

    def __init__(self, 
                 transforms, 
                 mode="train", 
                 fold = 0, 
                 imgsize=512,
                 crops_dir='jpeg',
                 data_path='data',
                 label_smoothing=0.01,
                 folds_csv='folds.csv.gz'):
        self.mode = mode
        self.fold = fold
        self.fold_csv = folds_csv
        self.crops_dir = crops_dir
        self.datadir = data_path
        self.data = self.loaddf()
        self.transform = transforms
        self.imgsize = imgsize
        self.label_cols = list(RSNAWEIGHTS.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        samp = self.data.loc[idx]
        img_name = self.image_file(samp)
                
        img = self.turboload(img_name)  
        if self.imgsize != 512:
            img = cv2.resize(img,(self.imgsize,self.imgsize), interpolation=cv2.INTER_LINEAR)
        
        if self.transform:       
            augmented = self.transform(image=img)
            img = augmented['image']   
        if self.mode == 'train':
            if 
            labels = torch.tensor(
                self.data.loc[idx, self.label_cols])
            return {'image': img, 'labels': labels}    
        else:      
            return {'image': img}
        
    def loaddf(self):
        fname = 'train.csv.zip' if self.mode in ['train', 'val'] else 'test.csv.zip'
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
        return os.path.join(self.datadir, 
                                self.crops_dir,
                                self.mode,
                                samp.StudyInstanceUID,
                                samp.SeriesInstanceUID,
                                samp.SOPInstanceUID) + '.jpg'



class nSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data, n = 4):
        self.data = data
        self.n = n
        self.sampler = self.sample(self.data)
                
    def __iter__(self):
        self.sampler = self.sample(self.data)
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)
    
    def sample(self, data):
        return data.sample(frac=1) \
                        .groupby("StudyInstanceUID") \
                        .head(self.n).index.tolist()