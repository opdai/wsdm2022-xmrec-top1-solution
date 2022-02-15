# coding:utf-8
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

SRC_PATH = '/home/workspace'
if not os.path.exists('/home/workspace/src/retrieval/lgcn_v2/data'):
    os.mkdir('/home/workspace/src/retrieval/lgcn_v2/data')
if not os.path.exists('/home/workspace/src/retrieval/lgcn_v2/test_result'):
    os.mkdir('/home/workspace/src/retrieval/lgcn_v2/test_result')


# S123的全部数据都加入训练集
def load_source_trainset(i=1):
    df_rating = pd.read_csv(f'{SRC_PATH}/DATA/s{i}/train.tsv', sep='\t', header=0)
    df_rating5 = pd.read_csv(f'{SRC_PATH}/DATA/s{i}/train_5core.tsv', sep='\t', header=0)
    df_qrel = pd.read_csv(f'{SRC_PATH}/DATA/s{i}/valid_qrel.tsv', sep='\t', header=0)
    print(f'  /DATA/s{i}:{df_rating.shape}, 5core:{df_rating5.shape}, qrel:{df_qrel.shape}')

    df = pd.concat([df_rating, df_rating5, df_qrel], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    print(f'[s{i}]df.shape:{df.shape}')
    return df


# 对T1保留T1qrel作为训练收敛指标参考，T2qrel加入训练
def load_target_trainset(i=1, keep_qrel=True):
    # 自己的train
    df_rating = pd.read_csv(f'{SRC_PATH}/DATA/t{i}/train.tsv', sep='\t', header=0)
    df_rating5 = pd.read_csv(f'{SRC_PATH}/DATA/t{i}/train_5core.tsv', sep='\t', header=0)
    # 加入另一个targetmarket的qrel
    another_i = 1 if i == 2 else 2
    df_another_qrel = pd.read_csv(f'{SRC_PATH}/DATA/t{another_i}/valid_qrel.tsv', sep='\t', header=0)
    append_list = [df_rating, df_rating5, df_another_qrel]
    # 是否加入自己的qrel
    if keep_qrel:
        print(f'  /DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}, t{another_i}valid:{df_another_qrel.shape}')
        print(f'  ->keep qrel as validation')
    else:
        df_qrel = pd.read_csv(f'{SRC_PATH}/DATA/t{i}/valid_qrel.tsv', sep='\t', header=0)
        append_list.append(df_qrel)
        print(f'  /DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}, t{another_i}valid:{df_another_qrel.shape}')
        print(f'  ->add its qrel:{df_qrel.shape} as trainset')

    df = pd.concat(append_list, axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    print(f'[t{i}]df.shape:{df.shape}')
    return df


def load_target_validset(i=1):
    df_qrel = pd.read_csv(f'{SRC_PATH}/DATA/t{i}/valid_qrel.tsv', sep='\t', header=0)
    df_qrel = df_qrel.drop_duplicates(['userId', 'itemId'])
    print(f'[t{i}valid]df.shape:{df_qrel.shape}')
    return df_qrel


def gather_vocabulary(df, key=''):
    _vocab = df[key].unique().tolist()
    _vocab = dict(zip(_vocab, range(len(_vocab))))
    print(f'[vocabulary]{key} size:{len(_vocab)}')
    return _vocab


def _helper_func(t):
    return ' '.join([str(ii) for ii in t])


if __name__ == '__main__':
    data_combination = {
        't1_s0',
        't1_s0_use_valid',
        't1_s1',
        't1_s1_use_valid',
        't1_s1s2',
        't1_s1s2_use_valid',
        't1_s1s2s3',
        't1_s1s2s3_use_valid',
        't1_s1s3',
        't1_s1s3_use_valid',
        't1_s2',
        't1_s2_use_valid',
        't1_s2s3',
        't1_s2s3_use_valid',
        't1_s3',
        't1_s3_use_valid',
        't2_s0',
        't2_s0_use_valid',
        't2_s1',
        't2_s1_use_valid',
        't2_s1s2',
        't2_s1s2_use_valid',
        't2_s1s2s3',
        't2_s1s2s3_use_valid',
        't2_s1s3',
        't2_s1s3_use_valid',
        't2_s2',
        't2_s2_use_valid',
        't2_s2s3',
        't2_s2s3_use_valid',
        't2_s3',
        't2_s3_use_valid',
        't1_s0__main',
        't1_s0__main_use_valid',
        't2_s0__main',
        't2_s0__main_use_valid',
    }

    for selected_data in data_combination:
        DATA_PATH = f'./data/{selected_data}'

        if not os.path.exists(DATA_PATH):
            print(f'->Create directory: {DATA_PATH}')
            os.mkdir(DATA_PATH)

        data_dic = {}
        modeltag = DATA_PATH.split('/')[-1]

        # setting
        modeltag_list = modeltag.split('_')

        # fst be target -> pickup valid
        target_market = modeltag_list[0]
        if target_market == 't1':
            data_dic['t1'] = load_target_trainset(i=1)
        elif target_market == 't2':
            data_dic['t2'] = load_target_trainset(i=2)
        else:
            raise Exception(f'error trainset selection with world.PATH:{modeltag}')

        data_dic['t1_valid'] = load_target_validset(i=1)
        data_dic['t2_valid'] = load_target_validset(i=2)

        # sec be train -> pickup train
        train_market = modeltag_list[1]
        for _mk in ['s1', 's2', 's3']:
            if _mk in train_market:
                data_dic[_mk] = load_source_trainset(i=int(_mk[-1]))

        print(f'->[dataloader] {DATA_PATH}\n'
              f'  Target:{target_market}\n'
              f'  Data:{list(data_dic.keys())}\n ')

        total_df = pd.concat(list(data_dic.values()), axis=0).drop_duplicates(['userId', 'itemId'])
        print(f'->total:{total_df.shape}')

        # vocabulary
        item_vocab = gather_vocabulary(total_df, key='itemId')
        user_vocab = gather_vocabulary(total_df, key='userId')

        savepath = f'{DATA_PATH}/item_list.txt'
        with open(savepath, 'w') as wf:
            wf.write('org_id remap_id\n')
            for kk, vv in item_vocab.items():
                wf.write(f'{kk} {vv}\n')

        savepath = f'{DATA_PATH}/user_list.txt'
        with open(savepath, 'w') as wf:
            wf.write('org_id remap_id\n')
            for kk, vv in user_vocab.items():
                wf.write(f'{kk} {vv}\n')

        # process
        for market_name, market_df in data_dic.items():
            print(f'->processing {market_name}..')

            # from userId to uidIndex
            market_df['itemId_idx'] = market_df['itemId'].apply(lambda t: item_vocab.get(t))
            market_df['userId'] = market_df['userId'].apply(lambda t: user_vocab.get(t))

            market_df = market_df.loc[:, ['userId', 'itemId_idx']]

            if '_valid' not in market_name:
                # from list to sequence
                market_df = market_df.groupby('userId').agg(list).reset_index()
                market_df['itemId_idx'] = market_df['itemId_idx'].apply(_helper_func)

            savepath = f'{DATA_PATH}/{market_name}.txt'
            print(f'->saving as {savepath}')
            with open(savepath, 'w') as wf:
                for i in market_df.values:
                    wf.write(f'{i[0]} {i[1]}\n')
