# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
import glob
import gc
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
from training.datasets.classifier_dataset import RSNASequenceDataset, collateseqfn
from training.zoo.sequence import SpatialDropout
from training.tools.utils import create_optimizer, AverageMeter
from training.losses import getLoss
from training import losses
from torch.optim.swa_utils import AveragedModel, SWALR
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
arg('--mixup_beta', type=float, default = 0.)
arg('--prefix', type=str, default='classifier_')
arg('--data-dir', type=str, default="data")
arg('--folds-csv', type=str, default='folds.csv.gz')
arg('--crops-dir', type=str, default='jpegip')
arg('--lstm_units',   type=int, default=2048)
arg('--epochs',   type=int, default=12)
arg('--nbags',   type=int, default=12)
arg('--label-smoothing', type=float, default=0.01)
arg('--logdir', type=str, default='logs/b2_1820')
arg("--local_rank", default=0, type=int)
arg("--seed", default=777, type=int)
args = parser.parse_args()

'''
if False:
    args.config = 'configs/b2.json'
    args.config = 'configs/b2_binary.json'
    args.config = 'configs/rnxt101_binary.json'
conf = load_config(args.config)
'''

embrgx = 'classifier_RSNAClassifier_resnext101_32x8d_*__fold*_epoch24__hflip*_transpose0_size320.emb'
datals = sorted(glob.glob(f'emb/{embrgx}*.pk'))
datals = [pd.read_pickle(f) for f in datals]
embls = [np.random.rand(f.shape[0], 2048).astype(np.float32) for f in datals]
datadf = pd.concat(datals, 0)
embmat = np.concatenate(embls, 0)
folddf = pd.read_csv(f'{args.data_dir}/{args.folds_csv}')
del embls
gc.collect()
# embls = [np.load(f.replace('.pk', '.npz'))['arr_0'] for f in datals]



'''
batch = []
for b in trndataset:
    batch.append(b)
    if len(batch)>7:
        break
'''

logger.info('Create traindatasets')
trndataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   folddf,
                                   mode="train",
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)
logger.info('Create valdatasets')
valdataset = RSNASequenceDataset(datadf, 
                                   embmat, 
                                   folddf,
                                   mode="valid",
                                   fold=args.fold,
                                   label_smoothing=args.label_smoothing,
                                   folds_csv=args.folds_csv)


logger.info('Create loaders...')
trnloader = DataLoader(trndataset, batch_size=args.batchsize, shuffle=True, num_workers=4, collate_fn=collateseqfn)
valloader = DataLoader(valdataset, batch_size=args.batchsize, shuffle=False, num_workers=4, collate_fn=collateseqfn)


batch = next(iter(trnloader))



# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet(nn.Module):
    def __init__(self, embed_size, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        
        self.embedding_dropout = SpatialDropout(0.0) #DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear = nn.Linear(LSTM_UNITS*2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

        output = self.linear(hidden)
        
        return output


logger.info('Create model')
model = NeuralNet(embmat.shape[0], LSTM_UNITS=args.lstm_units, DO = DROPOUT)
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = optim.Adam(plist, lr=lr)
scheduler = StepLR(optimizer, 1, gamma=lrgamma, last_epoch=-1)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")






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
    
if args.swa_epochs < 999:
    swa_start = conf['optimizer']['schedule']['epochs'] - args.swa_epochs
    logger.info(f'Swa start @ epoch {swa_start}')
    swa_model = AveragedModel(model)

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
    swaloader = DataLoader(trndataset, batch_size=args.batchsize, sampler = trnsampler, **loaderargs)
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
            if i % args.accum == 0:
                optimizer.step()
                optimizer.zero_grad()
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
            if i % args.accum == 0:
                optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        if i % args.accum == 0:
            optimizer.zero_grad()
        # if args.device != 'cpu': torch.cuda.synchronize()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, 
                          "loss": losses.avg, 'seen_prev': seenratio })
        
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if epoch > swa_start:
            swa_model.update_parameters(model)
        if i>30: break
        if i == max_iters - 1:
            break
    pbar.close()
    del sample, img, labels
    torch.cuda.empty_cache()
    if epoch > swa_start:
        swa_update_bn(trnloader, swa_model, args.device)

    if epoch > 0:
        seen = set(epoch_img_names[epoch]).intersection(
            set(itertools.chain(*[epoch_img_names[i] for i in range(epoch)])))
        seenratio = len(seen)/len(epoch_img_names[epoch])

    for idx, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)
    model = model.eval()
    
    # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
    # Update bn statistics for the swa_model at the end
    if epoch > swa_start:
        bce, acc, probdf = validate(swa_model, valloader, device = args.device)
    else:
        bce, acc, probdf = validate(model, valloader, device = args.device)

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
    torch.save({
        'epoch': current_epoch + 1,
        'state_dict': model.state_dict(),
        'bce_best': bce,
        }, args.output_dir + snapshot_name + f"_fold{args.fold}_epoch{current_epoch}")

    current_epoch += 1
