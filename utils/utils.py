# https://github.com/ildoonet/pytorch-gradual-warmup-lr
# https://github.com/PavelOstyakov/predictions_balancing/blob/master/run.py
import pickle
import argparse
import os
import torch
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict, OrderedDict
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import platform
import os
import gc
import glob
import pydicom
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import sys

RSNA_CFG = {
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


RSNAWEIGHTS = OrderedDict([('negative_exam_for_pe',  0.0736196319),
                ('indeterminate',	0.09202453988),
                ('chronic_pe'	, 0.1042944785),
                ('acute_and_chronic_pe',	0.1042944785),
                ('central_pe',	0.1877300613),
                ('leftsided_pe',	0.06257668712),
                ('rightsided_pe',	0.06257668712),
                ('rv_lv_ratio_gte_1',	0.2346625767), 
                ('rv_lv_ratio_lt_1',	0.0782208589)])


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def turbodump(f, img):
    # encoding BGR array to output.jpg with default settings.
    out_file = open(f, 'wb')
    out_file.write(jpeg.encode(img[:,:,::-1]))
    out_file.close()
    
# decoding input.jpg to BGR array
def turboload(f):
    in_file = open(f, 'rb')
    bgr_array = jpeg.decode(in_file.read())
    in_file.close()
    return bgr_array[:,:,::-1]

def get_first_of_dicom_field_as_int(x):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

# https://www.kaggle.com/redwankarimsony/rsna-str-pe-gradient-sigmoid-windowing
# https://radiopaedia.org/articles/windowing-ct
def window_image(img, window_center, window_width):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def bsb_window(img):
    lung_img1 = window_image(img, 40, 80)
    lung_img2 = window_image(img, 80, 200)
    lung_img3 = window_image(img, 600, 2000)
    
    bsb_img = np.zeros((lung_img1.shape[0], lung_img1.shape[1], 3))
    bsb_img[:, :, 0] = lung_img1
    bsb_img[:, :, 1] = lung_img2
    bsb_img[:, :, 2] = lung_img3
    bsb_img = (bsb_img*255).astype(np.uint8)
    return bsb_img

# https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930
def ip_window(img):
    '''
    RED channel / LUNG window / level=-600, width=1500
    GREEN channel / PE window / level=100, width=700
    BLUE channel / MEDIASTINAL window / level=40, width=400
    '''
    lung_img1 = window_image(img, -600, 1500)
    lung_img2 = window_image(img, 100, 700)
    lung_img3 = window_image(img, 40, 400)
    
    bsb_img = np.zeros((lung_img1.shape[0], lung_img1.shape[1], 3))
    bsb_img[:, :, 0] = lung_img1
    bsb_img[:, :, 1] = lung_img2
    bsb_img[:, :, 2] = lung_img3
    bsb_img = (bsb_img*255).astype(np.uint8)
    return bsb_img

def process_pixel_data(img_path):
    # Read dicom
    dicom_object = pydicom.dcmread(img_path)
    # Convert to jpeg and write to disk
    out_fname = img_path.replace(BASE_PATH, JPEG_PATH).replace('.dcm', '.jpg')
    fpath = Path(out_fname).parents[0]
    fpath.mkdir(parents=True, exist_ok=True)
    img = bsb_window(dicom_object)
    cv2.imwrite(out_fname, img)
            
            
def process_jpeg(img_path):
    # Read jpeg
    out_fname = img_path.replace(BASE_PATH, JPEG_PATH).replace('.dcm', '.jpg')
    img = turboload(out_fname)
    
def process_meta(img_path):
    dicom_object = pydicom.dcmread(img_path)
    window_center, window_width, intercept, slope = get_windowing(dicom_object)
    for col in meta_columns: 
        if col == 'WindowWidth':
            col_dict['WindowWidth'].append(window_width)
        elif col == 'WindowCenter':
            col_dict['WindowCenter'].append(window_center)
        elif col == 'RescaleIntercept':
            col_dict['RescaleIntercept'].append(intercept)
        elif col == 'RescaleSlope':
            col_dict['RescaleSlope'].append(slope)
        else:
            col_dict[col].append(str(getattr(dicom_object, col)))


