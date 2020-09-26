# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import sys
import itertools
from collections import defaultdict, OrderedDict
import platform
import glob
PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
os.chdir(PATH)
sys.path.append(PATH)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss
from utils.logs import get_logger
from utils.utils import RSNAWEIGHTS
from training.tools.config import load_config
import pandas as pd
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from tqdm import tqdm
from training.datasets.classifier_dataset import RSNAClassifierDataset, \
        nSampler, valSeedSampler, collatefn
from training.zoo import classifiers
from training.zoo.classifiers import swa_update_bn, validate

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
arg("--seed", default=777, type=int)
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default='weights/')
arg('--weightsrgx', type=str, default='classifier_RSNAClassifier_resnext101_32x8d_0__fold0_epoch2*')
arg('--epochs', type=str, default='21|22|23')
arg('--fold', type=int, default=0)
arg('--infer', type=bool, default=True)
arg('--emb', type=bool, default=False)
arg('--batchsize', type=int, default=4)
arg('--concatsteps', type=int, default=32)
arg('--labeltype', type=str, default='all') 
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
args = parser.parse_args()

HFLIP = False
TRANSPOSE = False

if False:
    args.config = 'configs/rnxt101_binary.json'
conf = load_config(args.config)

# Try using imagenet means
def create_val_transforms(size=300, HFLIPVAL = 1.0, TRANSPOSEVAL = 1.0):
    return A.Compose([
        A.Normalize(mean=conf['normalize']['mean'], 
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])

logger.info('Create valdatasets')
valdataset = RSNAClassifierDataset(mode="valid",
                                     fold=args.fold,
                                     crops_dir=args.crops_dir,
                                     classes = conf['classes'], 
                                     imgsize = conf['size'],
                                     data_path=args.data_dir,
                                     folds_csv=args.folds_csv,
                                     transforms=create_val_transforms(conf['size']))
loaderargs = {'num_workers' : 8, 
              'pin_memory': False, 
              'drop_last': False, 
              'collate_fn' : collatefn}
valsampler = valSeedSampler(valdataset.data, N = 5000, seed = args.seed)
valloader = DataLoader(valdataset, 
                       shuffle = False,
                       sampler = valsampler,
                       batch_size=args.batchsize, 
                       **loaderargs)

logger.info('Create model and optimisers')
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                              nclasses = len(conf['classes']),
                                              infer=False) 

wtls = glob.glob(f'{args.output_dir}/{args.weightsrgx}')
epochs = list(map(lambda x: f'_epoch{x}', args.epochs.split('|')))
wtls = [w for w in wtls if any(e in w for e in epochs)]
ckptls = [torch.load(wt, map_location=torch.device(args.device)) for wt in wtls]

for (w, checkpoint) in zip(wtls, ckptls):
    logger.info(f'Infer {w}')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)
    model = model.eval()
    bce, acc, probdf = validate(model, valloader, device = args.device, logger=logger)
    print("Weights {{w} Bce: {:.5f}, bce_best: {:.5f}".format(bce, bce_best))
    
'''
checkpoint = torch.load(f'{args.output_dir}/{args.weights}', 
                        map_location=torch.device(args.device))

model.load_state_dict(checkpoint['state_dict'])
model = model.to(args.device)
batch_size = conf['optimizer']['batch_size']

logger.info('Start inference')
imgnames = []
predls = []
model.eval()
pbar = tqdm(enumerate(valloader), total=len(valloader), desc="Weights {}".format(args.weights), ncols=0)
for i, sample in pbar:
    imgs = sample["image"].to(args.device)
    imgnames += sample['img_name']
    emb = model(imgs)
    embls.append(emb.detach().cpu().numpy().astype(np.float32))
outemb = np.concatenate(embls)

logger.info('Write embeddings : shape {} {}'.format(*outemb.shape))
fembname =  f'hflip{int(HFLIP)}_transpose{int(TRANSPOSE)}_size{conf["size"]}_fold{args.fold}__{args.weights}'
logger.info('Embedding file name : {}'.format(fembname))
np.savez_compressed(os.path.join('emb', fembname), outemb)
dumpobj(os.path.join(WORK_DIR, 'loader{logs/nohup_accum.out}'.format(HFLIP+TRANSPOSE, typ, SIZE, fold, epoch)), loader)
gc.collect()
'''
