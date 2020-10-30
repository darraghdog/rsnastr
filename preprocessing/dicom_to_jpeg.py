import platform
import os
import gc
import sys
import glob
import pydicom
import numpy as np # linear algebra
from pathlib import Path
import cv2
import gdcm
import zipfile
from io import StringIO
# conda install gdcm -c conda-forge
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--pkgpath', type=str, default='.')
arg('--infile', type=str, default='data/rsna-str-pulmonary-embolism-detection.zip')
arg('--jpegpath', type=str, default='data/jpegip')
args = parser.parse_args()

sys.path.append(args.pkgpath)

from training.tools.utils import get_logger

logger = get_logger('Preprocess', 'INFO') 
JPEG_PATH = f'{args.pkgpath}/{args.jpegpath}'

def get_first_of_dicom_field_as_int(x):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    # https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def window_image(img, window_center, window_width):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def ip_window(img):
    '''
    # https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930
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

def process_pixel_zip(filename):
    try:
        if 'dcm' in filename:
            with z.open(filename) as f:
                # Read dicom
                dicom_object = pydicom.read_file(f)
                # Convert to jpeg and write to disk
                if platform.system() == 'Darwin':
                    out_fname = filename.replace('dicom', JPEG_PATH).replace('.dcm', '.jpg')
                else:
                    out_fname = f'{JPEG_PATH}/{filename}'.replace('.dcm', '.jpg')
                fpath = Path(out_fname).parents[0]
                fpath.mkdir(parents=True, exist_ok=True)
                img = ip_window(dicom_object)
                cv2.imwrite(out_fname, img[:,:,::-1])
                logger.info(f'Success {filename}')
    except Exception as e: 
        logger.info(f'Failed {filename} : {e}')

def filekey(ls, ftype = '.jpg'):
    return [i.split('/')[-1].replace(ftype, '') for i in ls if ftype in i]

logger.info('Start extracting jpeg')
if args.infile == 'data/dicom.zip':
    z = zipfile.ZipFile(args.infile)     
    # Process train meta data
    with ThreadPoolExecutor() as threads:
       threads.map(process_pixel_zip, z.namelist())   
    gc.collect()
if args.infile == 'data/rsna-str-pulmonary-embolism-detection.zip':
    # May have to repeat this step a couple of times
    for i in range(10):
        z = zipfile.ZipFile(args.infile)
        jpgondisk = set(filekey(glob.glob(f'{JPEG_PATH}/*/*/*/*'), ftype = '.jpg'))
        zfiles = [z for z in z.namelist() if \
              (z.split('/')[-1].replace('.dcm', '') not in jpgondisk)]
        zfiles = [z for z in zfiles if '.dcm' in z]
        logger.info(f'There are {len(zfiles)} unprocessed files')
        if len(zfiles)==0: break
        # Process train meta data
        with ThreadPoolExecutor() as threads:threads.map(process_pixel_zip, zfiles)   
        gc.collect()

