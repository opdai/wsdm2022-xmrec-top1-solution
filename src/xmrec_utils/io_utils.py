import pandas as pd
from .config import DATA_BASEDIR
import os

def get_data_single(dirnm, basedir=DATA_BASEDIR):
    train = pd.read_csv(os.path.join(basedir,dirnm,'train.tsv'), sep='\t')
    train_5core = pd.read_csv(os.path.join(basedir,dirnm,'train_5core.tsv'), sep='\t')
    valid_qrel = pd.read_csv(os.path.join(basedir,dirnm,'valid_qrel.tsv'), sep='\t')
    valid_run = pd.read_csv(os.path.join(basedir,dirnm,'valid_run.tsv'), sep='\t',header=None,names=['userId','itemIds'])
    try:
        test_run = pd.read_csv(os.path.join(basedir,dirnm,'test_run.tsv'), sep='\t',header=None,names=['userId','itemIds'])
    except:
        test_run = None
        
    data_dict = {"train":train,
                 "train_5core":train_5core,
                 "valid_qrel":valid_qrel,
                 "valid_run":valid_run,
                 "test_run":test_run}
    return data_dict