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
from timm.models import resnext50_32x4d, resnext101_32x8d, resnext101_32x4d, mixnet_xxl
from timm.models import mixnet_xl, densenet169, densenet201, resnest200e
from timm.models import seresnext50_32x4d, seresnext101_32x4d, resnest200e, resnest269e
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
    "tf_efficientnet_b5_ns_04d_infer": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=False, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns_infer": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=False, drop_path_rate=0.2)
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
    "mixnet_xxl": {
        "features": 2048,
        "init_op": partial(mixnet_xxl, pretrained=True)
    },
    "mixnet_xl": {
        "features": 1536,
        "init_op": partial(mixnet_xl, pretrained=True)
    },
    "densenet201": {
        "features": 1920,
        "init_op": partial(densenet201, pretrained=True)
    },
    "densenet169": {
        "features": 1664,
        "init_op": partial(densenet169, pretrained=True)
    },
    "se101": {
        "features": 2048,
        "init_op": partial(seresnext101_32x4d, pretrained=True)
    },
    "sk50": {
        "features": 2048,
        "init_op": partial(skresnext50_32x4d, pretrained=True)
    },
    "resnest200e": {
        "features": 2048,
        "init_op": partial(resnest200e, pretrained=True)
    },
    "resnest269e": {
        "features": 2048,
        "init_op": partial(resnest269e, pretrained=True)
    }
}

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

