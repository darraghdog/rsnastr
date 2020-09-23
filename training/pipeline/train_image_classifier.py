# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import sys
import itertools
from collections import defaultdict, OrderedDict
import platform
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
from torch.cuda.amp import autocast


from tqdm import tqdm
import torch.distributed as dist
from training.datasets.classifier_dataset import RSNAClassifierDataset, \
        nSampler, valSeedSampler, collatefn
from training.zoo import classifiers
from training.tools.utils import create_optimizer, AverageMeter
from training.losses import getLoss
from training import losses

from tensorboardX import SummaryWriter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
logger = get_logger('Train', 'INFO') 

'''
aug = A.Compose([
        # A.HorizontalFlip(p=1.), right/left
        A.VerticalFlip(p=1.),
        A.Transpose(p=0.),
    ])
fname = 'data/jpeg/train/4f632056046b/03dbda10118a/53ccebd24e14.jpg'
img = cv2.imread(fname)[:,:,::-1]
img = cv2.resize(img, (360, 360))
from PIL import Image
Image.fromarray(img)
Image.fromarray(aug(image=img)['image'])
'''

logger.info('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
arg('--workers', type=int, default=6, help='number of cpu threads to use')
arg('--device', type=str, default='cpu' if platform.system() == 'Darwin' else 'cuda', help='device for model - cpu/gpu')
arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
arg('--output-dir', type=str, default='weights/')
arg('--resume', type=str, default='')
arg('--fold', type=int, default=0)
arg('--batchsize', type=int, default=4)
arg('--labeltype', type=str, default='all') # or 'single'
arg('--augextra', type=str, default=False) # or 'single'
arg('--mixup_beta', type=float, default = 0.)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
arg('--label-smoothing', type=float, default=0.01)
arg('--logdir', type=str, default='logs/b2_1820')
arg('--distributed', action='store_true', default=False)
arg('--freeze-epochs', type=int, default=0)
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
arg("--opt-level", default='O1', type=str)
arg("--test_every", type=int, default=1)
arg('--from-zero', action='store_true', default=False)
args = parser.parse_args()

if False:
    args.config = 'configs/b2.json'
    args.config = 'configs/b2_binary.json'
    args.config = 'configs/rnxt101_binary.json'
conf = load_config(args.config)

'''
from PIL import Image
img = cv2.imread('data/jpegip/train/6842db0937cf/51a8ec9ed5a8/09b37a7c0524.jpg')
Image.fromarray(img)

aug = create_train_transforms(size = img.shape[0])
augmented = aug(image=img)
img = augmented['image']
Image.fromarray(img)
'''


# Try using imagenet means
if not args.augextra:
    def create_train_transforms(size=300, distort = False):
        return A.Compose([
            #A.HorizontalFlip(p=0.5),   # right/left
            A.VerticalFlip(p=0.5), 
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, value = 0,
                                 rotate_limit=20, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            # A.Cutout(num_holes=40, max_h_size=size//7, max_w_size=size//7, fill_value=128, p=0.5), 
            #A.Transpose(p=0.5), # swing in -90 degrees
            A.Resize(size, size, p=1), 
            A.Normalize(mean=conf['normalize']['mean'], 
                        std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
            ToTensor()
        ])
else:
    def create_train_transforms(size=300, distort = False):
        return A.Compose([
            #A.HorizontalFlip(p=0.5),   # right/left
            A.VerticalFlip(p=0.5), 
            A.OneOf([
                A.RandomCrop(int(size*0.8), int(size*0.8), p = 0.5), 
                A.RandomCrop(int(size*0.9), int(size*0.9), p = 0.5), 
            ], p=1.0),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, value = 0,
                                 rotate_limit=20, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            # A.Cutout(num_holes=40, max_h_size=size//7, max_w_size=size//7, fill_value=128, p=0.5), 
            #A.Transpose(p=0.5), # swing in -90 degrees
            A.Resize(size, size, p=1), 
            A.Normalize(mean=conf['normalize']['mean'], 
                        std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
            ToTensor()
        ])

def create_val_transforms(size=300, HFLIPVAL = 1.0, TRANSPOSEVAL = 1.0):
    return A.Compose([
        #A.HorizontalFlip(p=HFLIPVAL),
        #A.Transpose(p=TRANSPOSEVAL),
        A.Normalize(mean=conf['normalize']['mean'], 
                    std=conf['normalize']['std'], max_pixel_value=255.0, p=1.0),
        ToTensor()
    ])

def validate(model, data_loader):
    probs = []#defaultdict(list)
    targets = []#defaultdict(list)
    studype = []
    img_names = []
    with torch.no_grad():
        for sample in tqdm(valloader):
            imgs = sample["image"].to(args.device)
            img_names += sample["img_name"]
            targets += sample["labels"].flatten().tolist()
            studype += sample['studype'].flatten().tolist()
            out = model(imgs)
            preds = torch.sigmoid(out).detach().cpu().numpy()
            probs.append(preds)
    probs = np.concatenate(probs, 0)
    targets = np.array(targets).round()
    studype = np.array(studype).round()
    negimg_idx = (targets < 0.5) & (studype > 0.5)
    posimg_idx = (targets > 0.5) & (studype > 0.5)
    negstd_idx = (targets < 0.5) & (studype < 0.5)

    negimg_loss = log_loss(targets[negimg_idx], probs[negimg_idx], labels=[0, 1])
    posimg_loss = log_loss(targets[posimg_idx], probs[posimg_idx], labels=[0, 1])
    negstd_loss = log_loss(targets[negstd_idx], probs[negstd_idx], labels=[0, 1])
    negimg_acc = (targets[negimg_idx] == (probs[negimg_idx] > 0.5).astype(np.int).flatten()).mean()
    posimg_acc = (targets[posimg_idx] == (probs[posimg_idx] > 0.5).astype(np.int).flatten()).mean()
    negstd_acc = (targets[negstd_idx] == (probs[negstd_idx] > 0.5).astype(np.int).flatten()).mean()
    avg_acc = (negimg_acc + posimg_acc + negstd_acc) / 3
    avg_loss= (negimg_loss + posimg_loss + negstd_loss) / 3
    log = f'Negimg PosStudy loss {negimg_loss:.4f} acc {negimg_acc:.4f}; '
    log += f'Posimg PosStudy loss {posimg_loss:.4f} acc {posimg_acc:.4f}; '
    log += f'Negimg NegStudy loss {negstd_loss:.4f} acc {negstd_acc:.4f}; '
    log += f'Avg 3 loss {avg_loss:.4f} acc {avg_acc:.4f}'
    logger.info(log)
    probdf = pd.DataFrame({'img': img_names, 
                           'label': targets.flatten(),
                           'studype': targets.flatten(),
                           'probs': probs.flatten()})
    return avg_loss, avg_acc, probdf

logger.info('Create traindatasets')
trndataset = RSNAClassifierDataset(mode="train",
                                       fold=args.fold,
                                       imgsize = conf['size'],
                                       crops_dir=args.crops_dir,
                                       classes = conf['classes'], 
                                       data_path=args.data_dir,
                                       label_smoothing=args.label_smoothing,
                                       folds_csv=args.folds_csv,
                                       transforms=create_train_transforms(conf['size']))
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
logger.info(50*'-')
logger.info(valdataset.data.loc[valsampler.sampler]['pe_present_on_image'].value_counts())
loaderargs = {'num_workers' : 8, 'pin_memory': False, 'drop_last': False, 'collate_fn' : collatefn}
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)

logger.info('Create model and optimisers')
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                              nclasses = len(conf['classes']) )
model = model.to(args.device)
reduction = "mean"

losstype = list(conf['losses'].keys())[0]
criterion = getLoss("BCEWithLogitsLoss", args.device)

optimizer, scheduler = create_optimizer(conf['optimizer'], model)
bce_best = 100
start_epoch = 0
batch_size = conf['optimizer']['batch_size']

os.makedirs(args.logdir, exist_ok=True)
summary_writer = SummaryWriter(args.logdir + '/' + conf.get("prefix", args.prefix) + conf['encoder'] + "_" + str(args.fold))

if args.from_zero:
    start_epoch = 0
current_epoch = start_epoch

if conf['fp16'] and args.device != 'cpu':
    scaler = torch.cuda.amp.GradScaler()
    
snapshot_name = "{}{}_{}_{}_".format(conf.get("prefix", args.prefix), conf['network'], conf['encoder'], args.fold)
max_epochs = conf['optimizer']['schedule']['epochs']

logger.info('Start training')
epoch_img_names = defaultdict(list)

'''
alldf = pd.read_csv('data/train.csv.zip')
allsampler = nSampler(alldf, pe_weight = 0.66, nmin = 2, nmax = 4, seed = None)
len(allsampler.sample(alldf)) * 0.8
'''
seenratio=0  # Ratio of seen in images in previous epochs

for epoch in range(start_epoch, max_epochs):
    '''
    Here we took out a load of things, check back 
    https://github.com/selimsef/dfdc_deepfake_challenge/blob/9925d95bc5d6545f462cbfb6e9f37c69fa07fde3/training/pipelines/train_classifier.py#L188-L201
    '''
    
    '''
    TRAIN
    '''
    ep_samps={'tot':0,'pos':0}
    losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    trnsampler = nSampler(trndataset.data, 
                          pe_weight = conf['pe_ratio'], 
                          nmin = conf['studynmin'], 
                          nmax = conf['studynmax'], 
                          seed = None)
    if current_epoch == 0: 
        trncts = trndataset.data.iloc[trnsampler.sample(trndataset.data)].pe_present_on_image.value_counts()
        valcts = valdataset.data.iloc[valsampler.sample(valdataset.data)].pe_present_on_image.value_counts()
        logger.info(f'Train class balance:\n{trncts}')
        logger.info(f'Valid class balance:\n{valcts}')
    trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = trnsampler, **loaderargs)
    model.train()
    pbar = tqdm(enumerate(trnloader), total=max_iters, desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "current_epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        epoch_img_names[current_epoch] += sample['img_name']
        imgs = sample["image"].to(args.device)
        # logger.info(f'Mean {imgs.mean()} std {imgs.std()} ')
        labels = sample["labels"].to(args.device).float()
        r = np.random.rand(1)
        if args.mixup_beta > 0:
            # generate mixed sample
            lam = np.random.beta(args.mixup_beta, args.mixup_beta)
            rand_index = torch.randperm(imgs.size()[0]).to(args.device)
            labels_a = labels
            labels_b = labels[rand_index]
            imgs = lam * imgs + (1 - lam) * imgs[rand_index]
            
        if conf['fp16'] and args.device != 'cpu':
            with autocast():
                out = model(imgs)
                if args.mixup_beta > 0:
                    loss = criterion(out, labels) * lam + \
                            criterion(out, labels) * (1. - lam)
                else:
                    loss = criterion(out, labels) # 0.6710
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            if args.mixup_beta > 0:
                loss = criterion(out, labels) * lam + \
                    criterion(out, labels) * (1. - lam)
            else:
                loss = criterion(out, labels)
                loss.backward()
            optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        optimizer.zero_grad()
        # if args.device != 'cpu': torch.cuda.synchronize()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, 
                          "loss": losses.avg, 'seen_prev': seenratio })
        
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i == max_iters - 1:
            break
    pbar.close()
    if epoch > 0:
        seen = set(epoch_img_names[epoch]).intersection(
            set(itertools.chain(*[epoch_img_names[i] for i in range(epoch)])))
        seenratio = len(seen)/len(epoch_img_names[epoch])

    for idx, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)
    model = model.eval()
    bce, acc, probdf = validate(model, valloader)

    if args.local_rank == 0:
        summary_writer.add_scalar('val/bce', float(bce), global_step=current_epoch)
        if bce < bce_best:
            print("Epoch {} improved from {:.5f} to {:.5f}".format(current_epoch, bce_best, bce))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'bce_best': bce,
                }, args.output_dir + snapshot_name + "_best_dice")
            bce_best = bce
            probdf.to_csv(args.output_dir + snapshot_name + "_best_probs.csv", index = False)
        print("Epoch: {} bce: {:.5f}, bce_best: {:.5f}".format(current_epoch, bce, bce_best))
    current_epoch += 1
