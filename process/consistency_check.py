# LIBRARIES
import os
import platform
import argparse
import pandas as pd
import numpy as np
import random


PATH = '/Users/dhanley/Documents/rsnastr' \
        if platform.system() == 'Darwin' else '/data/rsnastr'
os.chdir(PATH)
from utils.logs import get_logger

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg("--subfile", default='sub/submission.csv', type=str)
arg("--testfile", default='data/test.csv.zip', type=str)
args = parser.parse_args()

logger = get_logger('LSTM', 'INFO') 

sub = pd.read_csv(args.subfile)
test = pd.read_csv(args.testfile)
logger.info(f'Submission shape : {sub.shape}')
logger.info(f'Test shape : {test.shape}')


def clean_sub(sub, test):
    # Create the necessary meta data
    subtmp = sub.copy()
    subtmp['iid'] = subtmp.id.str.split('_').str[0]
    subtmp['Study'] = subtmp.id.str.split('_').str[0]
    subtmp = subtmp.merge(test[['StudyInstanceUID', 'SOPInstanceUID']], 
                          left_on = 'Study', right_on = 'SOPInstanceUID', how = 'left')
    subtmp.loc[~subtmp.StudyInstanceUID.isna(), 'Study'] = \
        subtmp.loc[~subtmp.StudyInstanceUID.isna(), 'StudyInstanceUID']
    subtmp = subtmp.drop(['StudyInstanceUID','SOPInstanceUID'], 1)
    
    # Identify positive and negative studies
    posStudy = subtmp[subtmp.id.str.contains('_negative_exam_for_pe')].query('label<0.5').Study.tolist()
    negStudy = subtmp[subtmp.id.str.contains('_negative_exam_for_pe')].query('label>=0.5').Study.tolist()
    
    # For negative studies clip the images at 0.49999
    iidix = subtmp.query('iid == id').set_index('Study').loc[negStudy].iid.tolist()
    subtmp = subtmp.set_index('iid')
    subtmp.label.loc[iidix] = subtmp.label.loc[iidix].clip(0, 0.49999)
    subtmp = subtmp.reset_index()
    
    # For positive studies raise the highest to 0.50001
    posdf = subtmp.query('iid == id').set_index('Study').loc[posStudy].reset_index()
    posidx = posdf.set_index('id').groupby('Study')['label'].idxmax().values
    subtmp = subtmp.set_index('iid')
    subtmp.label.loc[posidx] = subtmp.label.loc[posidx] .clip(0.50001, 1.0)
    subtmp = subtmp.reset_index()
    
    def subreshape(df, studyidx, regex):
        df = df.set_index('Study').loc[studyidx].reset_index()
        sidedf = df[df.id.str.contains(regex)][['id', 'label']]
        sidedf [['id','side']] = sidedf ["id"].str.split("_", 1, expand=True)
        sidedf = sidedf.pivot(index='id', columns='side', values='label')
        return sidedf
    
    
    # Address rule 2a
    sidedf = subreshape(subtmp, negStudy, regex='indeterminate|negative_exam')
    sidedf = sidedf[sidedf.max(1)<=0.5]
    sideStudy2aA = sidedf.idxmax(1).reset_index().agg('_'.join, axis=1).tolist()
    
    sidedf = subreshape(subtmp, negStudy, regex='indeterminate|negative_exam')
    sidedf = sidedf[sidedf.min(1)>0.5]
    sideStudy2aB = sidedf.idxmin(1).reset_index().agg('_'.join, axis=1).tolist()
    
    # Address rule 2b
    sidedf = subreshape(subtmp, negStudy, regex='rv_lv|central|sided|chronic')
    sidedf = sidedf[sidedf.max(1)>0.5]
    sideStudy2b = sidedf.idxmax(1).reset_index().agg('_'.join, axis=1).tolist()
    
    # Address rule 1aA
    sidedf = subreshape(subtmp, posStudy, regex='rv_lv_ratio')
    sidedf = sidedf[sidedf.max(1)<=0.5]
    sideStudy1aA = sidedf.idxmax(1).reset_index().agg('_'.join, axis=1).tolist()
    
    # Address rule 1aB
    sidedf = subreshape(subtmp, posStudy, regex='rv_lv_ratio')
    sidedf = sidedf[sidedf.min(1)>0.5]
    sideStudy1aB = sidedf.idxmin(1).reset_index().agg('_'.join, axis=1).tolist()
    
    # Address rule 1b
    sidedf = subreshape(subtmp, posStudy, regex='central_pe|sided_pe')
    sidedf = sidedf[sidedf.max(1)<=0.5]
    sideStudy1b = sidedf.idxmax(1).reset_index().agg('_'.join, axis=1).tolist()
    
    # Address rule 1c
    sidedf = subreshape(subtmp, posStudy, regex='chronic_pe')
    ind1cidx = sidedf[((sidedf>0.5).sum(1) == 2)].index.tolist()
    ind1cidx = sidedf.loc[ind1cidx].idxmin(1).reset_index().agg('_'.join, axis=1).tolist()
    
    # Address rule 1d
    sidedf = subreshape(subtmp, posStudy, regex='indeterminate|negative_exam_for_pe')
    ind1didx = sidedf[((sidedf>0.5).sum(1) == 2)].index.tolist()
    ind1didx = sidedf.loc[ind1didx].idxmin(1).reset_index().agg('_'.join, axis=1).tolist()
    

    '''
    Make changes
    '''
    subtmp = subtmp.set_index('id')
    idx = sideStudy1aA + sideStudy1b + sideStudy2aA
    subtmp.label.loc[idx] = subtmp.label.loc[idx].clip(0.5001, 1.0)
    idx = sideStudy1aB + ind1cidx + ind1didx + sideStudy2b + sideStudy2aB 
    subtmp.label.loc[idx] = subtmp.label.loc[idx].clip(0.0, 0.49999)
    subtmp = subtmp.reset_index()
    
    return subtmp[['id', 'label']]

def check_consistency(sub, test):
    
    '''
    Checks label consistency and returns the errors
    
    Args:
    sub   = submission dataframe (pandas)
    test  = test.csv dataframe (pandas)
    '''
    
    # EXAM LEVEL
    for i in test['StudyInstanceUID'].unique():
        df_tmp = sub.loc[sub.id.str.contains(i, regex = False)].reset_index(drop = True)
        df_tmp['StudyInstanceUID'] = df_tmp['id'].str.split('_').str[0]
        df_tmp['label_type']       = df_tmp['id'].str.split('_').str[1:].apply(lambda x: '_'.join(x))
        del df_tmp['id']
        if i == test['StudyInstanceUID'].unique()[0]:
            df = df_tmp.copy()
        else:
            df = pd.concat([df, df_tmp], axis = 0)
    df_exam = df.pivot(index = 'StudyInstanceUID', columns = 'label_type', values = 'label')
    
    # IMAGE LEVEL
    df_image = sub.loc[sub.id.isin(test.SOPInstanceUID)].reset_index(drop = True)
    df_image = df_image.merge(test, how = 'left', left_on = 'id', right_on = 'SOPInstanceUID')
    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace = True)
    del df_image['id']
    
    # MERGER
    df = df_exam.merge(df_image, how = 'left', on = 'StudyInstanceUID')
    ids    = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    labels = [c for c in df.columns if c not in ids]
    df = df[ids + labels]
    
    # SPLIT NEGATIVE AND POSITIVE EXAMS
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())
    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]
    
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'

    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'
    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'
    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS
    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'
    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    # OUTPUT
    print('Found', len(errors), 'inconsistent predictions')
    return errors

'''
# CHECK
errors = check_consistency(sub, test)
errors.broken_rule.value_counts()


# CHECK
sub1 = clean_sub(sub, test)
errors = check_consistency(sub1, test)
errors.broken_rule.value_counts()


(sub.label - sub1.label)[((sub.label - sub1.label)!=0)].round(1).value_counts().sort_index()
(sub.label - sub1.label)[((sub.label - sub1.label)!=0)].hist(bins = 100)
(abs((sub.label - sub1.label)[((sub.label - sub1.label)!=0)]) > 0.2).sum()
'''
