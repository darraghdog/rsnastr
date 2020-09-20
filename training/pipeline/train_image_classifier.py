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
from training.datasets.classifier_dataset import RSNAClassifierDataset, nSampler
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

# Data loaders

'''
Issues : 
    No such file or directory: 'data/jpegip/val/6897fa9de148/2bfbb7fd2e8b/c0f3cb036d06.jpg'
'''

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

# args.config = 'configs/b2.json'
# args.config = 'configs/b2_binary.json'
conf = load_config(args.config)

# Try using imagenet means
def create_train_transforms(size=300):
    return A.Compose([
        #A.HorizontalFlip(p=0.5),   # right/left
        A.VerticalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                             rotate_limit=20, p=0.5, border_mode = cv2.BORDER_REPLICATE),
        A.Cutout(num_holes=40, max_h_size=size//7, max_w_size=size//7, fill_value=128, p=0.5), 
        #A.Transpose(p=0.5), # swing in -90 degrees
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
valdataset = RSNAClassifierDataset(mode="val",
                                     fold=args.fold,
                                     crops_dir=args.crops_dir,
                                     classes = conf['classes'], 
                                     imgsize = conf['size'],
                                     data_path=args.data_dir,
                                     folds_csv=args.folds_csv,
                                     transforms=create_val_transforms(conf['size']))

valsampler = nSampler(valdataset.data, pe_weight = 0.66, nmin = 2, nmax = 4, seed = args.seed)
loaderargs = {'num_workers' : 8, 'pin_memory': False, 'drop_last': True}#, 'collate_fn' : collatefn}
valloader = DataLoader(valdataset, batch_size=args.batchsize, sampler = valsampler, **loaderargs)
logger.info('Create model and optimisers')
model = classifiers.__dict__[conf['network']](encoder=conf['encoder'], \
                                              nclasses = len(conf['classes']) )
model = model.to(args.device)
reduction = "mean"

losstype = list(conf['losses'].keys())[0]
weights = list(conf['losses'].values())[0]
loss = getLoss(losstype, torch.tensor(weights))
loss_functions = {"classifier_loss": loss}
loss_functions["classifier_loss"] = loss_functions["classifier_loss"].to(args.device)
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
    print("training epoch {} lr {:.7f}".format(current_epoch, scheduler.get_lr()[0]))
    trnsampler = nSampler(trndataset.data, pe_weight = 0.66, nmin = 2, nmax = 4, seed = None)
    cts = trndataset.data.iloc[trnsampler.sample(trndataset.data)].pe_present_on_image.value_counts()
    logger.info(f'Epoch class balance:\n{cts}')
    trnloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = trnsampler, **loaderargs)
    model.train()
    pbar = tqdm(enumerate(trnloader), total=max_iters, desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        #break
        epoch_img_names[epoch] += sample['img_name']
        imgs = sample["image"].to(args.device)
        # logger.info(f'Mean {imgs.mean()} std {imgs.std()} ')
        labels = sample["labels"].to(args.device).float()
        ep_samps['tot'] += imgs.shape[0]
        ep_samps['pos'] += labels[:,0].sum()
        balance = (ep_samps['pos']/ep_samps['tot']).item()
        if conf['fp16'] and args.device != 'cpu':
            with autocast():
                out = model(imgs)
                loss = loss_functions["classifier_loss"](labels, out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = loss_functions["classifier_loss"](labels, out)
            loss.backward()
            optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        optimizer.zero_grad()
        # if args.device != 'cpu': torch.cuda.synchronize()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, "loss": losses.avg,\
                "balance": balance, 'lossbatch': loss.item() })
        
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i == max_iters - 1:
            break
    pbar.close()
    if epoch > 0:
        seen = set(epoch_img_names[epoch]).intersection(
            set(itertools.chain(*[epoch_img_names[i] for i in range(epoch)])))
        seenratio = len(seen)/len(epoch_img_names[epoch])
        logger.info(f'Ratio seen in previous epochs : {seenratio}')

    for idx, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)
    model = model.eval()

    '''
    VALID
    '''

    probs = defaultdict(list)
    targets = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(valdataset):
            imgs = sample["image"].to(args.device)
            img_names = sample["img_name"]
            labels = sample["labels"].to(args.device).float()
            out = model(imgs)
            labels = labels.cpu().numpy()
            preds = torch.sigmoid(out).detach().cpu().numpy()
            for i in range(out.shape[0]):
                img_id = img_names[i]
                probs[img_id].append(preds[i].tolist())
                targets[img_id].append(labels[i].tolist())
    
    '''
    bce, probs, targets = validate(model, data_loader=data_val)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/bce', float(bce), global_step=current_epoch)
        if bce < bce_best:
            print("Epoch {} improved from {} to {}".format(current_epoch, bce_best, bce))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'bce_best': bce,
                }, args.output_dir + snapshot_name + "_best_dice")
            bce_best = bce
            with open("predictions_{}.json".format(args.fold), "w") as f:
                json.dump({"probs": probs, "targets": targets}, f)
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, args.output_dir + snapshot_name + "_last")
        print("Epoch: {} bce: {}, bce_best: {}".format(current_epoch, bce, bce_best))
        '''
        
    '''
    if args.local_rank == 0:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, args.output_dir + '/' + snapshot_name + "_last")
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, args.output_dir + snapshot_name + "_{}".format(current_epoch))
        if (epoch + 1) % args.test_every == 0:
            bce_best = evaluate_val(args, val_data_loader, bce_best, model,
                                    snapshot_name=snapshot_name,
                                    current_epoch=current_epoch,
                                    summary_writer=summary_writer)
    current_epoch += 1
    '''
    
    
    
    
    
    
    
    
