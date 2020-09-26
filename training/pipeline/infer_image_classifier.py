# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import sys
import gc
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
#from utils.swa_utils import swa
from utils.utils import RSNAWEIGHTS
from training.tools.config import load_config
import pandas as pd
import cv2
from utils.swa_utils import swa

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
arg('--runswa', default=False, type=lambda x: (str(x).lower() == 'true'))
arg('--infer', default=False, type=lambda x: (str(x).lower() == 'true'))
arg('--emb', default=False, type=lambda x: (str(x).lower() == 'true'))
arg('--batchsize', type=int, default=4)
arg('--concatsteps', type=int, default=32)
arg('--labeltype', type=str, default='all') 
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
args = parser.parse_args()

logger.info(args)
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
logger.info('Create traindatasets')
trndataset = RSNAClassifierDataset(mode="train",
                                       fold=args.fold,
                                       imgsize = conf['size'],
                                       crops_dir=args.crops_dir,
                                       classes = conf['classes'], 
                                       data_path=args.data_dir,
                                       label_smoothing=0.01,
                                       folds_csv=args.folds_csv,
                                       transforms=create_val_transforms(conf['size']))
logger.info('Create valdatasets')
valdataset = RSNAClassifierDataset(mode="valid",
                                     fold=args.fold,
                                     crops_dir=args.crops_dir,
                                     classes = conf['classes'], 
                                     imgsize = conf['size'],
                                     data_path=args.data_dir,
                                     folds_csv=args.folds_csv,
                                     transforms=create_val_transforms(conf['size']))

valsampler = valSeedSampler(valdataset.data, N = 5000, seed = args.seed)
trnsampler = nSampler(trndataset.data, 
                          pe_weight = conf['pe_ratio'], 
                          nmin = conf['studynmin'], 
                          nmax = conf['studynmax'], 
                          seed = None)
logger.info(50*'-')
logger.info(valdataset.data.loc[valsampler.sampler]['pe_present_on_image'].value_counts())
loaderargs = {'num_workers' : 8, 'pin_memory': False, 'drop_last': False, 'collate_fn' : collatefn}
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)
trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = trnsampler, **loaderargs)
logger.info('Create model and optimisers')
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                              nclasses = len(conf['classes']),
                                              infer=False) 


weightfiles = glob.glob(f'{args.output_dir}/{args.weightsrgx}')
epochs = list(map(lambda x: f'_epoch{x}', args.epochs.split('|')))
weightfiles = [w for w in weightfiles if any(e in w for e in epochs)]

if args.runswa:
    logger.info('Run SWA')
    net= swa(model, weightfiles, trnloader, args.batchsize//2, args.device)
    bce, acc, probdf = validate(net, valloader, device = args.device, logger=logger)
    print(f"SWA Bce: {bce:.5f}")

if args.infer:
    predls = []
    for f in weightfiles:
        model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                              nclasses = len(conf['classes']),
                                              infer=False)
        logger.info(f'Infer {f}')
        checkpoint = torch.load(f, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.half().to(args.device)
        model = model.eval()
        bce, acc, probdf = validate(model, valloader, device = args.device, logger=logger)
        print(f"Weights {f} Bce: {bce:.5f}")

if args.emb:
    valloader = DataLoader(valdataset, batch_size=args.batchsize, shuffle=False, **loaderargs)
    for f in weightfiles:
        logger.info(f'Infer {f}')
        model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                              nclasses = len(conf['classes']),
                                              infer=True)
        checkpoint = torch.load(f, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.half().to(args.device)
        model = model.eval()
        pbar = tqdm(enumerate(valloader), total=len(valloader), desc="Weights {}".format(f), ncols=0)
        embls = []
        with torch.no_grad():
            for i, sample in pbar:
                imgs = sample["image"].half().to(args.device)
                emb = model(imgs)
                embls.append(emb.detach().cpu().numpy().astype(np.float32))
        outemb = np.concatenate(embls)
        logger.info('Write embeddings : shape {} {}'.format(*outemb.shape))
        fembname =  f'{f}__hflip{int(HFLIP)}_transpose{int(TRANSPOSE)}_size{conf["size"]}.emb'
        fembname = fembname.replace(args.output_dir, '')
        logger.info('Embedding file name : {}'.format(fembname))
        np.savez_compressed(os.path.join('emb', fembname), outemb)
        valdataset.data.to_pickle( f'emb/{fembname}.pk' )
        gc.collect()
    
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
