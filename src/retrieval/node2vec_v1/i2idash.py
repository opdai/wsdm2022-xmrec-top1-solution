# coding:utf-8
import os
import json
import resource
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import networkx as nx

from time import time
from datetime import datetime, timedelta
from pytz import timezone
from itertools import combinations
from zipfile import ZipFile
from gensim.models import Word2Vec

sys.path.insert(0, '/home/workspace/')
sys.path.insert(1, 'graphemb')

from graphemb.model.deepwalk import DeepWalk
from graphemb.model.node2vec import Node2Vec
from gensim.models.callbacks import CallbackAny2Vec
import zipfile
import math
from tqdm import tqdm
from src.botbase import Cache
from i2iv1Params import walk_length,num_walks,workers,_p,_q,vector_size,window,sg,hs,negative

UTC8 = timezone('Asia/Shanghai')


def time_now():
    return datetime.now(UTC8).strftime('%Y%m%d%H%M%S')


def load_source_trainset(i=1):
    df_rating = pd.read_csv(f'/home/workspace/DATA/s{i}/train.tsv', sep='\t', header=0)
    df_rating5 = pd.read_csv(f'/home/workspace/DATA/s{i}/train_5core.tsv', sep='\t', header=0)
    df_qrel = pd.read_csv(f'/home/workspace/DATA/s{i}/valid_qrel.tsv', sep='\t', header=0)
    print(f'/DATA/s{i}:{df_rating.shape}, 5core:{df_rating5.shape}, qrel:{df_qrel.shape}')

    df = pd.concat([df_rating, df_rating5, df_qrel], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    print(f'\t[i=={i}]df.shape:{df.shape}')
    return df


def load_target_trainset(i=1):
    df_rating = pd.read_csv(f'/home/workspace/DATA/t{i}/train.tsv', sep='\t', header=0)
    df_rating5 = pd.read_csv(f'/home/workspace/DATA/t{i}/train_5core.tsv', sep='\t', header=0)
    df_qrel = pd.read_csv(f'/home/workspace/DATA/t{i}/valid_qrel.tsv', sep='\t', header=0)
    print(f'/DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}, qrel:{df_qrel.shape}')

    df = pd.concat([df_rating, df_rating5, df_qrel], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    print(f'\t[i=={i}]df.shape:{df.shape}')
    return df


def gather_vocabulary(df):
    itemid_vocab = df['itemId'].unique().tolist()
    itemid_vocab = dict(zip(itemid_vocab, range(len(itemid_vocab))))
    print(f'\titemId vocabulary size:{len(itemid_vocab)}')
    return itemid_vocab


def trans_vocab_index(df, id_vocab):
    df['itemId_idx'] = df['itemId'].apply(lambda t: id_vocab.get(t))
    return df


def gen_user_sequence(df):
    total_edge_set = set()
    df_sequence = df.groupby('userId')['itemId_idx'].agg(list).reset_index()['itemId_idx'].values
    print(f'\tsequence.shape:{df_sequence.shape}')

    for seq in df_sequence:
        pair_list = list(combinations(seq, 2))
        total_edge_set.update(pair_list)
    print(f'\tedges.shape:{len(total_edge_set)}')
    return total_edge_set


def load_target_valid_test(i=1):
    df_test_run = pd.read_csv(f'/home/workspace/DATA/t{i}/test_run.tsv', sep='\t', header=None, names=['userId', 'itemId'])
    df_valid_run = pd.read_csv(f'/home/workspace/DATA/t{i}/valid_run.tsv', sep='\t', header=None, names=['userId', 'itemId'])
    print(f'DATA/t{i}/test_run:{df_test_run.shape}, /DATA/t{i}/valid_run:{df_valid_run.shape}')

    # split candidate list into rows
    df_test_run['itemId'] = df_test_run['itemId'].apply(lambda t: t.split(','))
    df_valid_run['itemId'] = df_valid_run['itemId'].apply(lambda t: t.split(','))

    df_test_run = df_test_run.explode('itemId')
    df_valid_run = df_valid_run.explode('itemId')
    return df_valid_run, df_test_run


def _restore_score(t):
    return t['user_recall_dic'].get(t['itemId'], -10)


def formup_submittion_bigraph(_src_df, _uid2recalldic, savename='default'):
    # recall baseon user sequence
    _src_df['user_recall_dic'] = _src_df['userId'].apply(lambda t: _uid2recalldic.get(t, {}))
    # recall score by i2i similarity dic
    _src_df['score'] = _src_df.apply(_restore_score, axis=1)
    # formatting
    _submit = _src_df.loc[:, ['userId', 'itemId', 'score']].sort_values(['userId', 'score'], ascending=False)

    _submit.to_csv(savename, sep='\t', index=False)
    print(f'->saving as {savename}:{_submit.shape}')


def get_sim_item(df, user_col, item_col):
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    sim_item = {}
    # only taking trainset items into concern
    for item, users in tqdm(item_user_dict.items()):

        sim_item.setdefault(item, {})
        for u in users:

            tmp_len = len(user_item_dict[u])
            # then calc item similarity within interacted user's sequence
            for relate_item in user_item_dict[u]:
                sim_item[item].setdefault(relate_item, 0)
                sim_item[item][relate_item] += 1 / (math.log(len(users) + 1) * math.log(tmp_len + 1))

    return sim_item, user_item_dict


def zip_submittion(srcpath, outputpath):
    fzip = zipfile.ZipFile(outputpath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirs, fnames in os.walk(srcpath):
        fpath = path.replace(srcpath, '')
        for fname in fnames:
            fzip.write(os.path.join(path, fname), os.path.join(fpath, fname))
    fzip.close()


def main():
    t0 = time()
    s1_df = load_source_trainset(i=1)
    s2_df = load_source_trainset(i=2)
    s3_df = load_source_trainset(i=3)
    t1_df = load_target_trainset(i=1)
    t2_df = load_target_trainset(i=2)
    total_df = pd.concat([s1_df, s2_df, s3_df, t1_df, t2_df]).drop_duplicates(['userId', 'itemId'], keep='last')
    print(f'[{time_now()}]all trainset shape:{total_df.shape}')

    t12_df = pd.read_csv(f'/home/workspace/DATA/t1t2/valid_qrel.tsv', sep='\t', header=0)
    total_df = pd.concat([total_df, t12_df]).drop_duplicates(['userId', 'itemId'], keep='last')
    print(f'[{time_now()}]add t1t2/valid_qrel.tsv into trainset shape:{total_df.shape}')

    # itemId vocabulary
    itemvoc = gather_vocabulary(total_df)
    with open(f"./vocabulary.txt", 'w') as wf:
        for kk, vv in itemvoc.items():
            wf.write(f'{kk}\t{vv}\n')

    # transform itemId to integer index
    df_idx = trans_vocab_index(total_df, itemvoc)
    print(df_idx)
    t1 = time()
    print(f'[{time_now()}]transform vocabulary index: {(t1 - t0):.2f}secs')

    # gather behavior sequence
    total_edge_set = gen_user_sequence(df_idx)
    t2 = time()
    print(f'[{time_now()}]generate user behavior sequence: {(t2 - t1):.2f}secs')

    # nondirect graph
    graph = nx.Graph()
    total_edge_set = list(set([tuple(sorted(ii)) for ii in total_edge_set]))
    graph.add_edges_from(list(total_edge_set))
    t3 = time()
    print(f'[{time_now()}]generate graph: {(t3 - t2):.2f}secs '
          f'nodes:{len(list(graph.nodes()))} edges:{len(total_edge_set)}')

    # graph walker
    walk_model = Node2Vec(
        graph,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=_p,
        q=_q,
        use_rejection_sampling=1)
    t4 = time()
    print(f'[{time_now()}]sentence walker: {(t4 - t3):.2f}secs '
          f'nodes:{len(list(graph.nodes()))} edges:{len(total_edge_set)} '
          f'walk_length:{walk_length} num_walks:{num_walks} p:{_p} q:{_q} '
          f'graphtype:{"BFS" if _q > 1 else "DFS"}')

    edges_save_path = f"./edges_p{_p}_q{_q}_len{walk_length}_" \
                      f"{time_now()}.txt"
    with open(edges_save_path, 'w') as wf:
        for ii in walk_model.sentences:
            wf.write(f'{ii}\n')

    # word2vector
    print(f'[{time_now()}]w2v training...'
          f'\tparams:p={_p} q={_q} sg={sg} hs={hs} neg={negative} window={window}')
    w2v_model = Word2Vec(
        sentences=walk_model.sentences,
        vector_size=vector_size,  # embeddingsize
        window=window,  # 句子滑窗大小
        min_count=0,  # 过滤最小词频
        workers=64,  # 进程数
        sg=sg,  # 1：skipgram；0：cbow
        hs=hs,  # 1：hierarchical softmax；0：negative sampling
        negative=negative,  # negative samples
        epochs=3,  # iteration
    )
    print(f'[{time_now()}]w2v training consumption: {(time() - t4):.2f}secs')

    # embedding saving
    save_path = f"./xmarket_p{_p}_q{_q}_sg{sg}_hs{hs}_neg{negative}_window{window}_" \
                f"{time_now()}.w2v.model"
    w2v_model.save(save_path)
    print(f'[{time_now()}]w2vmodel saved as: {save_path}')

    # predict
    # w2vmodel = Word2Vec.load("./xmarket_p20_q13_sg1_hs0_neg10_window50_20220118232020.w2v.model")
    itemvoc = pd.read_csv('./vocabulary.txt', sep='\t', header=None, names=['itemId', 'item_idx'])

    item_voc = dict(zip(itemvoc['itemId'], itemvoc['item_idx']))
    item_list = itemvoc['itemId'].tolist()

    df_t1_valid, df_t1_test = load_target_valid_test(1)
    df_t2_valid, df_t2_test = load_target_valid_test(2)

    # get user sequence
    _, u2i_dict = get_sim_item(total_df, 'userId', 'itemId')

    # always check if uid notin predict userlist
    pred_user_list = pd.read_csv(f'/home/workspace/DATA/t1t2/valid_qrel.tsv', sep='\t', header=0)['userId'].unique().tolist()

    # calc recall dic for every user in userlist
    uid2recalldic = {}
    topn = 200
    for _uid in tqdm(pred_user_list):
        _global_recall_dict = {}
        _user_sequence = u2i_dict.get(_uid, {})
        for _item in _user_sequence:
            try:
                # trans2idx
                _item_idx = item_voc[_item]
            except:
                print(f'item:{_item} not in vocabulary')
                continue

            # i2i sim recall by w2v
            _local_rec_dic = w2v_model.wv.most_similar(_item_idx, topn=topn)
            for _k, _v in _local_rec_dic:
                # back2itemId
                _k = item_list[_k]
                if _k in _global_recall_dict:
                    _global_recall_dict[_k] += _v
                # if _k in _global_recall_dict and _v > _global_recall_dict[_k]:
                #     _global_recall_dict[_k] = _v
                else:
                    _global_recall_dict[_k] = _v
        uid2recalldic[_uid] = _global_recall_dict
    print(f'done recall for {len(uid2recalldic)} user in testset')

    if not os.path.exists('/home/workspace/OUTPUT/00_NEW/'):
        print(f'->Create directory: /home/workspace/OUTPUT/00_NEW/')
        os.mkdir('/home/workspace/OUTPUT/00_NEW/')
    spath = '/home/workspace/OUTPUT/00_NEW/i2ibfssubmit'
    if not os.path.exists(spath):
        print(f'->Create directory: {spath}')
        os.mkdir(f'{spath}')
        os.mkdir(f'{spath}/t1')
        os.mkdir(f'{spath}/t2')
    formup_submittion_bigraph(df_t1_valid, uid2recalldic, savename=f'{spath}/t1/valid_pred.tsv')
    formup_submittion_bigraph(df_t1_test, uid2recalldic, savename=f'{spath}/t1/test_pred.tsv')
    formup_submittion_bigraph(df_t2_valid, uid2recalldic, savename=f'{spath}/t2/valid_pred.tsv')
    formup_submittion_bigraph(df_t2_test, uid2recalldic, savename=f'{spath}/t2/test_pred.tsv')

    ddf = pd.DataFrame(uid2recalldic.items())
    ddf = ddf.rename(columns={0: 'userId', 1: 'itemscoredic'})

    ddf_ex_itemid = pd.DataFrame(ddf['itemscoredic'].apply(lambda t: list(t.keys())))
    ddf_ex_score = pd.DataFrame(ddf['itemscoredic'].apply(lambda t: list(t.values())))

    global_score = pd.concat([
        ddf,
        ddf_ex_itemid.explode('itemscoredic').rename(columns={'itemscoredic': 'itemId'}),
        ddf_ex_score.explode('itemscoredic').rename(columns={'itemscoredic': 'score'})], axis=1)

    if not os.path.exists('/home/workspace/OUTPUT/00_NEW/'):
        print(f'->Create directory: /home/workspace/OUTPUT/00_NEW/')
        os.mkdir('/home/workspace/OUTPUT/00_NEW/')
    spath = '/home/workspace/OUTPUT/00_NEW/i2ibfssubmit'
    if not os.path.exists(spath):
        print(f'->Create directory: {spath}')
        os.mkdir(f'{spath}')
        os.mkdir(f'{spath}/t1')
        os.mkdir(f'{spath}/t2')
    # global_score.to_csv(_save_path, header=True, index=False, seq='\t')
    Cache.dump_pkl(global_score, f'{spath}/t1/test_pred_all.pkl')
    Cache.dump_pkl(global_score, f'{spath}/t2/test_pred_all.pkl')


if __name__ == "__main__":
    main()
