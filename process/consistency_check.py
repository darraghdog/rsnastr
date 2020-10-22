# LIBRARIES
import os
import platform
import argparse
import pandas as pd
import numpy as np

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



def clean_sub(sub):
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
    subtmp[subtmp.id.str.contains('_negative_exam_for_pe')].hist(bins = 100)
    posStudy = subtmp[subtmp.id.str.contains('_negative_exam_for_pe')].query('label<0.5').Study.tolist()
    negStudy = subtmp[subtmp.id.str.contains('_negative_exam_for_pe')].query('label>0.5').Study.tolist()
    subtmp.query('iid == id').groupby('Study')['label'].max().loc[negStudy].hist(bins = 500)
    subtmp.query('iid == id').groupby('Study')['label'].max().loc[posStudy].hist(bins = 500)
    
    iidix = subtmp.query('iid == id').set_index('Study').loc[negStudy].iid.tolist()
    subtmp = subtmp.set_index('iid')
    subtmp.label.loc[iidix] = subtmp.label.loc[iidix].clip(0, 0.49999)
    subtmp.label.loc[iidix] .hist()
    subtmp = subtmp.reset_index()
    
    #ix = subtmp.id.str.contains('_negative_exam_for_pe')
    #posStudy = subtmp[ix].query('label < 0.5').Study.values
    #negStudy = subtmp[ix].query('label >= 0.5').Study.values
    
    
    # Addres rule 1a
    ix = subtmp.Study.isin(posStudy)
    
    rvlv = subtmp.set_index('Study').loc[posStudy]
    rvlvg = rvlv[rvlv.id.str.contains('rv_lv_ratio_g')]
    rvlvl = rvlv[rvlv.id.str.contains('rv_lv_ratio_l')]
    rvlvg['rvlvl_label'] = rvlvl.label
    rvlvg['rvlvg_label'] = rvlvg.label
    
    rvlvg['ctoverhalf'] = (rvlvg.filter(like='rvlv')>0.5).sum(1)
    rvlvg['nearesttohalf'] = abs(rvlvg.filter(like='rvlv') - 0.5).idxmin(1)
    
    # Raise these Study's rv_lv_ratio_gte_1 values to 0.501
    rule1apartA = [ f'{v}_rv_lv_ratio_gte_1' for v in  
                   rvlvg.query('ctoverhalf == 0').query("nearesttohalf =='rvlvg_label'").index.values]
    # Raise these Study's with rv_lv_ratio_lt_1 values to 0.501
    rule1apartB = [f'{v}_rv_lv_ratio_lt_1' for v in 
                   rvlvg.query('ctoverhalf == 0').query("nearesttohalf =='rvlvl_label'").index.values]
    # Drop these Study's rv_lv_ratio_gte_1 values to 0.499
    rule1apartC = [f'{v}_rv_lv_ratio_gte_1' for v in 
                   rvlvg.query('ctoverhalf == 2').query("nearesttohalf =='rvlvg_label'").index.values]
    # Drop these Study's  rv_lv_ratio_lt_1 values to 0.499
    rule1apartD = [f'{v}_rv_lv_ratio_lt_1' for v in 
                   rvlvg.query('ctoverhalf == 2').query("nearesttohalf =='rvlvl_label'").index.values]
    
    # Addres rule 1b
    sidedf = subtmp.set_index('Study').loc[posStudy].reset_index()
    sidedf = sidedf[sidedf.id.str.contains('central_pe|sided_pe')][['id', 'label']]
    sidedf [['id','side']] = sidedf ["id"].str.split("_", 1, expand=True)
    sidedf = sidedf.pivot(index='id', columns='side', values='label')
    sidedf = sidedf[sidedf.max(1)<0.5]
    # Raise these to 0.50001
    sideStudy = sidedf.idxmax(1).reset_index().agg('_'.join, axis=1).tolist()
    
    
    '''
    Make changes
    '''
    subtmp = subtmp.set_index('id')
    
    subtmp.label.loc[rule1apartA + rule1apartB + sideStudy] = 0.50001
    subtmp.label.loc[rule1apartC + rule1apartD] = 0.49999
    
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


# CHECK
errors = check_consistency(sub, test)
errors.broken_rule.value_counts()

# CHECK
sub1 = clean_sub(sub)
errors = check_consistency(sub1, test)
errors.head()
errors.broken_rule.value_counts()

sub1.to_csv('~/Downloads/sub1.csv', index = False)
sub.to_csv('~/Downloads/sub.csv', index = False)


