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
from pytorch_transformers.modeling_bert import BertConfig, BertEncoder


class TransformerNet(nn.Module):
    def __init__(self, 
                 cfg, 
                 nimgclasses = 1, 
                 nstudyclasses = 9):
        super(TransformerNet, self).__init__()
        
        self.nimgclasses = nimgclasses
        self.nstudyclasses = nstudyclasses
        self.cfg = cfg

        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            max_position_embeddings=cfg.max_position_embeddings, 
            intermediate_size=cfg.intermediate_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        
        self.encoder = BertEncoder(self.config).to(cfg.device)

        self.img_linear_out = nn.Linear(cfg.hidden_size, self.nimgclasses)
        self.study_linear_out = nn.Linear(cfg.hidden_size, self.nstudyclasses)
        
    def extended_mask(self, mask):
        # Prep mask
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.cfg.nlayers
        
        return extended_attention_mask, head_mask
        
    def forward(self, x, mask, lengths=None):
        
        extended_attention_mask, head_mask =  extended_mask( mask )
        
        # Pass thru encoder
        encoded_layers = encoder(x, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        
        # Pass output to a linear layer
        img_output = self.img_linear_out(sequence_output).squeeze()
        study_output = self.study_linear_out(sequence_output[:, -1]).squeeze()
        
        return study_output, img_output


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
                 infer = False, 
                 DO = 0.3):
        super(LSTMNet, self).__init__()
        
        self.nimgclasses = nimgclasses
        self.nstudyclasses = nstudyclasses
        self.embed_size = embed_size
        self.embedding_dropout = SpatialDropout(DO)
        self.infer = infer
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.img_linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.img_linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.study_linear1 = nn.Linear(LSTM_UNITS*4, LSTM_UNITS*4)

        self.img_linear_out = nn.Linear(LSTM_UNITS*2, self.nimgclasses)
        self.study_linear_out = nn.Linear(LSTM_UNITS*4, self.nstudyclasses)

    def forward(self, x, mask, lengths=None):
        
        h_embedding = x

        #h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        #h_embadd = self.embedding_dropout(h_embadd)
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # Masked mean and max pool for study level prediction
        #avg_pool = torch.sum(h_lstm2, 1) * (1/ mask.sum(1)).unsqueeze(1)
        #max_pool, _ = torch.max(h_lstm2, 1)
        if self.infer:
            mask = mask.float()
        avg_pool = torch.sum(h_lstm2 * mask.unsqueeze(-1), 1)* \
                             (1/ mask.sum(1)).unsqueeze(1)
        max_pool, _ = torch.max(h_lstm2 * mask.unsqueeze(-1), 1)

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
                            padding = 2,
                            kernel_size = 5)
        self.cnn1db = nn.Conv1d(self.dense_units, 
                            self.dense_units,
                            padding = 2,
                            kernel_size = 5)
        self.linear = nn.Linear(self.dense_units, self.dense_units)
        self.embedding_dropout = SpatialDropout(dropout)
        self.linear_img_out = nn.Linear(self.dense_units, 1)
        self.linear_exm = nn.Linear(self.dense_units*2, self.dense_units*2)
        self.linear_exm_out = nn.Linear(self.dense_units*2, 9)
        '''
        self.linear_out = nn.Linear(self.dense_units*2, 1)
        '''
    
    def forward(self, x):
        # Input is batch of image sequences
        batch_size, seqlen = x.size()[:2]
        # Flatten to make a single long list of frames
        x1 = x.view(batch_size * seqlen, *x.size()[2:])
        # Pass each frame thru SPPNet
        emb = self.encoder(x1)
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
        
        # Classifier Image
        hidden = embc + h_pool_linear 
        outimg = self.linear_img_out(hidden)
        
        # Classifier Exam
        avg_pool = torch.mean(embc, 1)
        max_pool, _ = torch.max(embc, 1)
        h_study_conc = torch.cat((max_pool, avg_pool), 1)
        h_study_conc_linear1  = nn.functional.relu(self.linear_exm(h_study_conc))
        study_hidden = h_study_conc + h_study_conc_linear1
        outexm = self.linear_exm_out(study_hidden)
        
        return outimg, outexm

