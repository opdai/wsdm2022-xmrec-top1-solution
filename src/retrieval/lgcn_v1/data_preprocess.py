import argparse
import pandas as pd

import os
from os import path
import json
import resource
import sys
import pickle
from zipfile import ZipFile
from pprint import pprint

DATA_PATH = './Data/s123t12_withoutqrel'
if not os.path.exists('/home/workspace/src/retrieval/lgcn_v1/Data'):
    os.mkdir('/home/workspace/src/retrieval/lgcn_v1/Data')


def load_source_trainset(i=1):
    df_rating = pd.read_csv('/home/workspace/DATA/s{}/train.tsv'.format(i), sep='\t', header=0)
    df_rating5 = pd.read_csv('/home/workspace/DATA/s{}/train_5core.tsv'.format(i), sep='\t', header=0)
    df_qrel = pd.read_csv('/home/workspace/DATA/s{}/valid_qrel.tsv'.format(i), sep='\t', header=0)
    # print(f'/DATA/s{i}:{df_rating.shape}, 5core:{df_rating5.shape}, qrel:{df_qrel.shape}')

    df = pd.concat([df_rating, df_rating5, df_qrel], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    # print(f'\t[i=={i}]df.shape:{df.shape}')
    return df


def load_target_trainset(i=1, keep_qrel=True):
    # train
    df_rating = pd.read_csv('/home/workspace/DATA/t{}/train.tsv'.format(i), sep='\t', header=0)
    df_rating5 = pd.read_csv('/home/workspace/DATA/t{}/train_5core.tsv'.format(i), sep='\t', header=0)
    append_list = [df_rating, df_rating5]
    # qrel
    if keep_qrel:
        # print(
        #     f'/DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}')
        print('->keep qrel as validation')
    else:
        df_qrel = pd.read_csv('/home/workspace/DATA/t{}/valid_qrel.tsv'.format(i), sep='\t', header=0)
        append_list.append(df_qrel)
        # print(
        #     f'/DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}')
        print('->add its qrel as trainset')

    df = pd.concat(append_list, axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    # print(f'\t[i=={i}]df.shape:{df.shape}')
    return df


def gather_vocabulary(df):
    itemid_vocab = df['itemId'].unique().tolist()
    itemid_vocab = dict(zip(itemid_vocab, range(len(itemid_vocab))))
    print('\titemId vocabulary size:{}'.format(len(itemid_vocab)))
    return itemid_vocab


def trans_vocab_index(df, id_vocab):
    df['itemId_idx'] = df['itemId'].apply(lambda t: id_vocab.get(t))
    return df


def _helper_func(t):
    return ' '.join([str(ii) for ii in t])


def data_preprocess4lgcn():
    df_train_t1 = load_target_trainset(1, keep_qrel=True)
    df_train_t2 = load_target_trainset(2, keep_qrel=True)

    df_train_s1 = load_source_trainset(1)
    df_train_s2 = load_source_trainset(2)
    df_train_s3 = load_source_trainset(3)

    total_df = pd.concat([
        df_train_t1,
        df_train_t2,
        df_train_s1,
        df_train_s2,
        df_train_s3,
    ]).drop_duplicates(['userId', 'itemId'], keep='last')
    print(total_df.shape)

    item_vocab = gather_vocabulary(total_df)
    savepath = DATA_PATH + '/item_list.txt'
    with open(savepath, 'w') as wf:
        wf.write('org_id remap_id\n')
        for kk, vv in item_vocab.items():
            wf.write('{} {}\n'.format(kk, vv))

    df_idx = trans_vocab_index(total_df, item_vocab)
    df_idx = df_idx.groupby('userId').agg(list).reset_index()

    df_idx_sequence = df_idx.loc[:, ['userId', 'itemId_idx']]
    df_idx_sequence['itemId_idx'] = df_idx_sequence['itemId_idx'].apply(_helper_func)

    user_vocab = df_idx_sequence['userId'].unique().tolist()
    user_vocab = dict(zip(user_vocab, range(len(user_vocab))))
    print('\tuserId vocabulary size:{}'.format(len(user_vocab)))

    savepath = DATA_PATH + '/user_list.txt'
    with open(savepath, 'w') as wf:
        wf.write('org_id remap_id\n')
        for kk, vv in user_vocab.items():
            wf.write('{} {}\n'.format(kk, vv))

    df_idx_sequence['userId'] = df_idx_sequence['userId'].apply(lambda t: user_vocab.get(t))

    qrel_df = pd.read_csv('/home/workspace/DATA/t1t2/valid_qrel.tsv', sep='\t', header=0)
    # print(f'total qrel_df:{qrel_df.shape}')
    qrel_df = qrel_df[qrel_df['userId'].isin(user_vocab)]
    # print(f'filteruser qrel_df:{qrel_df.shape}')
    qrel_df = qrel_df[qrel_df['itemId'].isin(item_vocab)]
    # print(f'filteritem qrel_df:{qrel_df.shape}')

    df_idx_qrel = trans_vocab_index(qrel_df, item_vocab)
    df_idx_qrel = df_idx_qrel.groupby('userId').agg(list).reset_index()
    df_idx_qrel = df_idx_qrel.loc[:, ['userId', 'itemId_idx']]
    df_idx_qrel['itemId_idx'] = df_idx_qrel['itemId_idx'].apply(_helper_func)
    df_idx_qrel['userId'] = df_idx_qrel['userId'].apply(lambda t: user_vocab.get(t))

    # split train/test set
    train = df_idx_sequence
    test = df_idx_qrel
    savepath = DATA_PATH + '/train.txt'
    with open(savepath, 'w') as wf:
        for i in train.values:
            wf.write('{} {}\n'.format(i[0], i[1]))
    print('->trainset saved as {}'.format(savepath))

    savepath = DATA_PATH + '/test.txt'
    with open(savepath, 'w') as wf:
        for i in test.values:
            wf.write('{} {}\n'.format(i[0], i[1]))
    print('->testset saved as {}'.format(savepath))

    return item_vocab, user_vocab


if __name__ == '__main__':

    if not os.path.exists(DATA_PATH):
        print('->Create directory: {}'.format(DATA_PATH))
        os.mkdir(DATA_PATH)

    data_preprocess4lgcn()
