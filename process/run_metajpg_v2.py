# https://www.kaggle.com/teeyee314/rsna-pe-metadata-with-multithreading
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
from turbojpeg import TurboJPEG
# import gdcm
import zipfile
from io import StringIO
# conda install gdcm -c conda-forge
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
os.chdir(PATH)
from utils.logs import get_logger
logger = get_logger('Create folds', 'INFO') 

TYPE = 'ip' # 'bsb'
BASE_PATH = f'{PATH}/data'
JPEG_PATH = f'{PATH}/data/jpeg{TYPE}'
try:
    jpeg = TurboJPEG()
except:
    logger.info('Failed to load turbojpeg')
logger.info(f'Home path list \n{os.listdir(BASE_PATH )}')

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


def process_pixel_zip(filename):
    if 'dcm' in filename:
        with z.open(filename) as f:
            # Read dicom
            dicom_object = pydicom.dcmread(f)
            # Convert to jpeg and write to disk
            if platform.system() == 'Darwin':
                out_fname = filename.replace('dicom', JPEG_PATH).replace('.dcm', '.jpg')
            else:
                out_fname = f'{JPEG_PATH}/{filename}'.replace('.dcm', '.jpg')
            fpath = Path(out_fname).parents[0]
            fpath.mkdir(parents=True, exist_ok=True)
            if TYPE == 'bsb':
                img = bsb_window(dicom_object)
            if TYPE == 'ip':
                img = ip_window(dicom_object)
            cv2.imwrite(out_fname, img[:,:,::-1])
            
            
#filename = 'dicom/train/0dcb58a69743/59598f169a63/1911bab9f095.dcm'

logger.info('Start extracting jpeg')
if platform.system() == 'Darwin':
    z = zipfile.ZipFile(f'{BASE_PATH}/dicom.zip')     
    # Process train meta data
    with ThreadPoolExecutor() as threads:
       threads.map(process_pixel_zip, z.namelist())   
    gc.collect()
else:
    z = zipfile.ZipFile(f'{BASE_PATH}/rsna-str-pulmonary-embolism-detection.zip')    
    # Process train meta data
    with ThreadPoolExecutor() as threads:
       threads.map(process_pixel_zip, z.namelist())   
    gc.collect()

'''
imgip = cv2.imread('/Users/dhanley/Downloads/tmp.jpg')[:,:,::-1]
Image.fromarray(imgip)
Image.fromarray(img)
'''


