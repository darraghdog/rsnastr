import os
import platform
PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
os.chdir(PATH)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, 'scripts')
from utils.logs import get_logger
logger = get_logger('Create folds', 'INFO') # noqa


DATAPATH = 'data'
trndf = pd.read_csv(os.path.join(DATAPATH, 'raw/train.csv.zip'))
# tstdf = pd.read_csv(os.path.join(DATAPATH, 'raw/test.csv.zip'))

logger.info('Create folds')
folddf = pd.DataFrame({ 'StudyInstanceUID': trndf['StudyInstanceUID'].unique()})
folddf['fold'] = (folddf.index.values)%5

# Write out 
folddf.to_csv(f'{DATAPATH}/folds.csv.gz', compression='gzip', index = False)
