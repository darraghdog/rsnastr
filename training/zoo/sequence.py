# Ripped from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/zoo/classifiers.py
from functools import partial

import timm
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch.nn.functional as F
from torch import nn

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class LSTMNet(nn.Module):
    def __init__(self, 
                 embed_size, 
                 nimgclasses = 1, 
                 nstudyclasses = 9, 
                 LSTM_UNITS=64, 
                 DO = 0.3):
        super(LSTMNet, self).__init__()
        
        self.nimgclasses = nimgclasses
        self.nstudyclasses = nstudyclasses
        self.embed_size = embed_size
        self.embedding_dropout = SpatialDropout(0.0) #DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.img_linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.img_linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.study_linear1 = nn.Linear(LSTM_UNITS*4, LSTM_UNITS*4)

        self.img_linear_out = nn.Linear(LSTM_UNITS*2, self.nimgclasses)
        self.study_linear_out = nn.Linear(LSTM_UNITS*4, self.nstudyclasses)

    def forward(self, x, mask, lengths=None):
        
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # Masked mean and max pool for study level prediction
        avg_pool = torch.sum(h_lstm2, 1) * (1/ mask.sum(1)).unsqueeze(1)
        max_pool, _ = torch.max(h_lstm2, 1)
        
        # Get study level prediction
        h_study_conc = torch.cat((max_pool, avg_pool), 1)
        h_study_conc_linear1  = nn.functional.relu(self.study_linear1(h_study_conc))
        study_hidden = h_study_conc + h_study_conc_linear1
        study_output = self.study_linear_out(study_hidden)
        
        # Get study level prediction
        h_img_conc_linear1  = nn.functional.relu(self.img_linear1(h_lstm1))
        h_img_conc_linear2  = nn.functional.relu(self.img_linear2(h_lstm2))
        img_hidden = h_lstm1 + h_lstm2 + h_img_conc_linear1 + h_img_conc_linear2 # + h_embadd
        img_output = self.img_linear_out(img_hidden)
        
        return study_output, img_output

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class StudyImgNet(nn.Module):
    def __init__(self, encoder = 'mixnet_m', dropout = 0.2, 
                 nclasses = 10, dense_units = 512):
        # Only resnet is supported in this version
        super(StudyImgNet, self).__init__()
        self.encoder = timm.create_model(encoder, 
                                         pretrained=True, 
                                         num_classes=0)
        self.dense_units = dense_units
        self.dropout = dropout
        self.embed_size = self.encoder.num_features
        self.cnn1da = nn.Conv1d(self.embed_size, 
                            self.dense_units,
                            padding = 1,
                            kernel_size = 5)
        self.cnn1db = nn.Conv1d(self.dense_units, 
                            self.dense_units,
                            padding = 1,
                            kernel_size = 5)
        self.linear = nn.Linear(self.dense_units, self.dense_units)
        self.embedding_dropout = SpatialDropout(dropout)
        self.linear_out = nn.Linear(self.dense_units, nclasses)
        '''
        self.linear_out = nn.Linear(self.dense_units*2, 1)
        '''
    
    def forward(self, x):
        # Input is batch of image sequences
        batch_size, seqlen = x.size()[:2]
        # Flatten to make a single long list of frames
        x = x.view(batch_size * seqlen, *x.size()[2:])
        # Pass each frame thru SPPNet
        emb = self.encoder(x)
        # Split back out to batch
        emb = emb.view(batch_size, seqlen, emb.size()[1])
        emb = self.embedding_dropout(emb)
        
        # Pass batch thru sequential model(s)
        embc = self.cnn1da(emb.transpose(2, 1))
        embc = F.relu(embc)
        embc = self.embedding_dropout(embc)
        embc = self.cnn1db(embc)
        embc = F.relu(embc)
        embc = embc.transpose(2, 1)
        h_pool_linear = F.relu(self.linear(embc))
        
        # Classifier
        hidden = embc + h_pool_linear 
        out = self.linear_out(hidden)
        return out