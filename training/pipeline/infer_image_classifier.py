# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import pickle
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
from training.zoo.sequence import StudyImgNet
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
arg('--type', type=str, default='image', help='Image model of study model')
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
valdataset = RSNAClassifierDataset(mode="valid",
                                         fold=args.fold,
                                         crops_dir=args.crops_dir,
                                         imgclasses=conf["image_target_cols"],
                                         studyclasses=conf['exam_target_cols'],
                                         imgsize = conf['size'],
                                         data_path=args.data_dir,
                                         folds_csv=args.folds_csv,
                                         transforms=create_val_transforms(conf['size']))
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
valsampler = valSeedSampler(valdataset.data, N = 5000, seed = args.seed)
logger.info(50*'-')
logger.info(valdataset.data.loc[valsampler.sampler]['pe_present_on_image'].value_counts())
loaderargs = {'num_workers' : 16, 'pin_memory': False, 'drop_last': False, 'collate_fn' : collatefn}
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)
allloader = DataLoader(alldataset, batch_size=args.batchsize, shuffle=False, **loaderargs)

weightfiles = glob.glob(f'{args.output_dir}/{args.weightsrgx}')
epochs = list(map(lambda x: f'_epoch{x}', args.epochs.split('|')))
#weightfiles = [w for w in weightfiles if any(e in w for e in epochs)]
logger.info(f'Weights to process: {weightfiles}')

if args.emb:
    for f in weightfiles:
        logger.info(f'Infer {f}')
        if args.type=='image':
            model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                                  nclasses = len(conf['classes']),
                                                  infer=True)
            checkpoint = torch.load(f, map_location=torch.device(args.device))
            model.load_state_dict(checkpoint['state_dict'])
        if args.type=='study':
            nc = len(conf['image_target_cols']+conf['exam_target_cols'])
            model =StudyImgNet(conf['encoder'], 
                               dropout = 0.0,
                               nclasses = nc,
                               dense_units = 512)
            checkpoint = torch.load(f, map_location=torch.device(args.device))
            model.load_state_dict(checkpoint)
            model = model.encoder
        model = model.half().to(args.device)
        model = model.eval()
        logger.info(f'Embeddings total : {len(allloader)}')
        pbar = tqdm(enumerate(allloader), total=len(allloader), desc="Weights {}".format(f), ncols=0)
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
        valdataset.data.to_pickle( f'emb/{fembname}.data.pk' )
        with open(f'emb/{fembname}.imgnames.pk', 'wb') as handle:
            pickle.dump(img_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()