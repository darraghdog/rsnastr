# Ripped from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/zoo/classifiers.py
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
from timm.models import skresnext50_32x4d
from timm.models.dpn import dpn92, dpn131
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
from timm.models.senet import seresnext50_32x4d, seresnext101_32x4d
from timm.models import resnext50_32x4d, resnext101_32x8d, resnext101_32x4d
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

encoder_params = {
    "dpn92": {
        "features": 2688,
        "init_op": partial(dpn92, pretrained=True)
    },
    "dpn131": {
        "features": 2688,
        "init_op": partial(dpn131, pretrained=True)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns_03d": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_03d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_04d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)
    },
    "resnext50_32x4d": {
        "features": 2048,
        "init_op": partial(resnext50_32x4d, pretrained=True)
    },
    "resnext101_32x8d": {
        "features": 2048,
        "init_op": partial(resnext101_32x8d, pretrained=True)
    },
    "resnext101_32x4d": {
        "features": 2048,
        "init_op": partial(resnext101_32x4d, pretrained=True)
    },
    "se50": {
        "features": 2048,
        "init_op": partial(seresnext50_32x4d, pretrained=True)
    },
    "mixnet_xl": {
        "features": 2048,
        "init_op": partial(mixnet_xl, pretrained=True)
    },
    "resnest200e": {
        "features": 2048,
        "init_op": partial(resnest200e, pretrained=True)
    },
    "se101": {
        "features": 2048,
        "init_op": partial(seresnext101_32x4d, pretrained=True)
    },
    "sk50": {
        "features": 2048,
        "init_op": partial(skresnext50_32x4d, pretrained=True)
    },
}




class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x

'''
class RSNAClassifier(nn.Module):
    def __init__(self, encoder, nclasses, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], nclasses)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
'''

class RSNAClassifier(nn.Module):
    def __init__(self, encoder, nclasses, dropout_rate=0.0, infer = False) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], nclasses)
        self.infer = infer 

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        if self.infer:
            return x
        x = self.dropout(x)
        x = self.fc(x)
        return x



class RSNAClassifierGWAP(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = GlobalWeightedAvgPool2d(encoder_params[encoder]["features"])
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    
def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 1., -2., 1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 2., -4., 2., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-2., 8., -12., 8., -2.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv


class RSNAClassifierSRM(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.srm_conv = setup_srm_layer(3)
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        noise = self.srm_conv(x)
        x = self.encoder.forward_features(noise)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def validate(model, data_loader, device, logger):
    probs = []#defaultdict(list)
    targets = []#defaultdict(list)
    studype = []
    img_names = []
    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader)):
            imgs = sample["image"].half().to(device)
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
    negimg_acc = (targets[negimg_idx] == (probs[negimg_idx] > 0.5).astype(np.int).flatten()).mean()
    posimg_loss = log_loss(targets[posimg_idx], probs[posimg_idx], labels=[0, 1])
    posimg_acc = (targets[posimg_idx] == (probs[posimg_idx] > 0.5).astype(np.int).flatten()).mean()
    negstd_loss = log_loss(targets[negstd_idx], probs[negstd_idx], labels=[0, 1])
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


def swa_update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model) 
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it 
        is assumed that :meth:`model.forward()` should be called on the first 
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for sample in loader:
        '''
        if isinstance(input, (list, tuple)):
            input = input[0]
        '''
        input = sample['image'] 
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
