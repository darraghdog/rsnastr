# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import sys
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import albumentations as A
from albumentations.pytorch import ToTensor
from collections import defaultdict, OrderedDict
import platform
PATH = '/Users/dhanley/Documents/kaggle/rsnastr' \
        if platform.system() == 'Darwin' else '/mount'
os.chdir(PATH)
sys.path.append(PATH)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss
from training.tools.utils import get_logger
from training.tools.config import load_config
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from training.datasets.dataset import RSNAClassifierDataset, valSeedSampler, collatefn, nSampler
from training.zoo import classifiers
from training.tools.utils import create_optimizer, AverageMeter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logger = get_logger('Dump weights', 'INFO') 

logger.info('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=8, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--resume', type=str, default='')
arg('--fold', type=int, default=0)
arg('--accum', type=int, default=1)
arg('--batchsize', type=int, default=4)
arg('--labeltype', type=str, default='all') # or 'single'
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
arg('--label-smoothing', type=float, default=0.01)
arg('--weights', type=str, default='classifier__RSNAClassifier_tf_efficientnet_b5_ns_04d_fold0_img512_accum1___best')
args = parser.parse_args()

if False:
    args.config = 'configs/effnetb5_lr5e4_multi.json'
conf = load_config(args.config)
logger.info(conf)

# Try using imagenet means
def create_val_transforms(size=300, HFLIPVAL = 1.0, TRANSPOSEVAL = 1.0):
    return A.Compose([
        A.Normalize(mean=conf['normalize']['mean'], 
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])

logger.info('Create datasets')
alldataset = RSNAClassifierDataset(mode="all",
                                           fold=args.fold,
                                           imgsize = conf['size'],
                                           crops_dir=args.crops_dir,
                                           imgclasses=conf["image_target_cols"],
                                           studyclasses=conf['exam_target_cols'],
                                           data_path=args.data_dir,
                                           label_smoothing=0.00,
                                           folds_csv=args.folds_csv,
                                           transforms=create_val_transforms(conf['size']))
logger.info(50*'-')
loaderargs = {'num_workers' : args.workers, 'pin_memory': False, 'drop_last': False, 'collate_fn' : collatefn}
allloader = DataLoader(alldataset, batch_size=args.batchsize, shuffle=False, **loaderargs)

weightfile = f'weights/{args.weights}'
logger.info(f'Weights to process: {weightfile}')
nclasses = len(conf['image_target_cols']) + len(conf['exam_target_cols'] )
model = classifiers.__dict__[conf['network']](encoder=f"{conf['encoder']}_infer", \
                                      nclasses = nclasses,
                                      infer=True)
checkpoint = torch.load(weightfile, map_location=torch.device(args.device))
model.load_state_dict(checkpoint['state_dict'])

model = model.half().to(args.device)
model = model.eval()
logger.info(f'Embeddings total : {len(alldataset)}; total batches : {len(allloader)}')
pbar = tqdm(enumerate(allloader), total=len(allloader), desc="Weights {}".format(weightfile), ncols=0)
embls = []
img_names = []
with torch.no_grad():
    for i, sample in pbar:
        img_names += sample['img_name']
        imgs = sample["image"].half().to(args.device)
        emb = model(imgs)
        embls.append(emb.detach().cpu().numpy().astype(np.float32))
outemb = np.concatenate(embls)
logger.info('Write embeddings : shape {} {}'.format(*outemb.shape))
fembname =  f'{f}__all_size{conf["size"]}.emb'
#fembname = 'emb/'+fembname.replace(args.output_dir, '')
logger.info('Embedding file name : {}'.format(fembname))
np.savez_compressed(os.path.join('emb', fembname), outemb)
with open(f'emb/{fembname}.imgnames.pk', 'wb') as handle:
    pickle.dump(img_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
gc.collect()
