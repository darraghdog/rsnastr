# https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/pipelines/train_classifier.py
import argparse
import json
import os
from collections import defaultdict, OrderedDict
import platform
PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
os.chdir(PATH)

from sklearn.metrics import log_loss
from utils.logs import get_logger
from utils.utils import RSNAWEIGHTS
from training.tools.config import load_config
import pandas as pd
import cv2
from utils.utils import RSNAWEIGHTS
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


trndf = pd.read_csv('data/train.csv.zip')



trndf.iloc[0]
cols = ['pe_present_on_image'] + list(RSNAWEIGHTS.keys())

print(cols)

nunqdf = trndf.groupby('StudyInstanceUID')[cols].nunique()

trndf.query('StudyInstanceUID == "6897fa9de148"').iloc[:, :8]

nunqdf.loc['6897fa9de148']
nunqdf.min(0)