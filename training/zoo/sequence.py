# Ripped from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/zoo/classifiers.py
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

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
    def __init__(self, embed_size, LSTM_UNITS=64, DO = 0.3):
        super(LSTMNet, self).__init__()
        
        self.embed_size = embed_size
        self.embedding_dropout = SpatialDropout(0.0) #DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.img_linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.img_linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.study_linear1 = nn.Linear(LSTM_UNITS*4, LSTM_UNITS*4)

        self.img_linear_out = nn.Linear(LSTM_UNITS*2, 1)
        self.study_linear_out = nn.Linear(LSTM_UNITS*4, 1)

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
