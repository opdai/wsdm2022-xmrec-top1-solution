import sys

sys.path.insert(0, '/home/workspace')
import os
import gc
import math
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.botbase import mkdirs, bash2py
import pandas as pd
from src.xmrec_utils.io_utils import get_data_single
from src.xmrec_utils.config import OUTPUT_BASEDIR
from itertools import combinations, permutations
from src.botbase import Cache
from src.botbase import get_n_parts_idx_lst
from concurrent.futures import ProcessPoolExecutor, as_completed


def xLogX(x):
    x = x + 1e-7
    return x * math.log(x)


def entropy(*elements):
    # print(elements)
    sum_all = 0
    result = 0.0
    for element in elements:
        result += xLogX(element)
        sum_all += element
    return xLogX(sum_all) - result


def llr(k11, k12, k21, k22):
    rowEntropy = entropy(k11 + k12, k21 + k22)
    columnEntropy = entropy(k11 + k21, k12 + k22)
    matrixEntropy = entropy(k11, k12, k21, k22)
    if rowEntropy + columnEntropy < matrixEntropy:
        return 0.0
    return 2.0 * (rowEntropy + columnEntropy - matrixEntropy)


def entropy2(*elements):
    sum_all = sum(elements)
    result = 0.0
    for element in elements:
        iszero = 1e-7  # 1 if element == 1 else 0
        result += element * math.log((element + iszero) / sum_all)
    return -result


def llr2(k11, k12, k21, k22):
    rowEntropy = entropy2(k11, k12) + entropy2(k21, k22)
    columnEntropy = entropy2(k11, k21) + entropy2(k12, k22)
    matrixEntropy = entropy2(k11, k12, k21, k22)
    print(rowEntropy, columnEntropy, matrixEntropy)
    if (rowEntropy + columnEntropy > matrixEntropy):
        return 0.0
    return 2 * (matrixEntropy - rowEntropy - columnEntropy)


class llr_i2i:
    def __init__(self, train_data):
        self.train_data = train_data
        self.u2i_dict = defaultdict(set)
        self.i2u_dict = defaultdict(set)
        for _, row in tqdm(self.train_data.iterrows()):
            self.u2i_dict[row["userId"]].add(row["itemId"])
            self.i2u_dict[row["itemId"]].add(row["userId"])
        self.all_users = set(self.u2i_dict.keys())
        self.item_sim_dict = defaultdict(dict)

    def _build_i2i_single(self, ibatch):
        item_sim_dict = defaultdict(dict)
        for i, j in ibatch:
            if i in item_sim_dict[j] and isinstance(item_sim_dict[j][i],
                                                    (int, float, np.number)):
                item_sim_dict[i][j] = item_sim_dict[j][i]
                continue
            a = self.i2u_dict[i]
            b = self.i2u_dict[j]
            k11, k12, k21, k22 = len(a & b), len(a - b), len(b - a), len(
                self.all_users) - len(a.union(b))
            item_sim_dict[i][j] = llr(k11, k12, k21, k22)
            item_sim_dict[j][i] = llr(k11, k12, k21, k22)
        return item_sim_dict

    def build_i2i_similarity_map(self):
        item_pairs = list(combinations(self.i2u_dict.keys(), 2))
        print("item pairs length：{}".format(len(item_pairs)))
        self.item_sim_dict = self._build_i2i_single(item_pairs)
        self.built = True

    def build_i2i_similarity_map_parallel(self):
        item_pairs = list(combinations(self.i2u_dict.keys(), 2))
        print("item pairs length：{}".format(len(item_pairs)))
        item_pairs_lst = get_n_parts_idx_lst(item_pairs, 1000)
        N_WORKERS = 30
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_train_key = {}
            for idx, ibatch in enumerate(item_pairs_lst):
                future = executor.submit(self._build_i2i_single, ibatch=ibatch)
                future_to_train_key[future] = "IDX_" + str(idx)
            for future in as_completed(future_to_train_key):
                train_key = future_to_train_key[future]
                try:
                    item_sim_dict = future.result()
                    self.item_sim_dict.update(item_sim_dict)
                except Exception as exc:
                    print(f'{train_key} generated an exception: {exc}')
        self.built = True

    def recommend(self, user_id):
        rank = defaultdict(int)
        try:
            interacted_items = self.u2i_dict[user_id]
        except:
            interacted_items = {}
        for i in interacted_items:  # 历史交互列表
            for j, wij in sorted(self.item_sim_dict[i].items(),
                                 key=lambda d: d[1],
                                 reverse=True):  # 大->小
                rank[j] += wij
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)

    def predict(self, test_data):
        if not self.built:
            print("build i2i similarity map...")
            self.build_i2i_similarity_map()
        test_uid_pool = test_data['userId'].unique()
        # 进行召回
        recom_item = []
        for uid in tqdm(test_uid_pool):
            rank_item = self.recommend(uid)
            for j in rank_item:
                recom_item.append([uid, j[0], j[1]])
        recom_item_df = pd.DataFrame(recom_item)
        recom_item_df.columns = ['userId', 'itemId', 'score']
        return recom_item_df


def getDCG(scores):
    return np.sum(np.divide(
        np.power(2, scores) - 1,
        np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list],
                             dtype=np.float32)  # 生成IDCG的rankscore
    # idcg = getDCG(relevance)
    idcg = 1
    dcg = getDCG(rank_scores)
    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg


def match_func(items1, items2):
    res = []
    for it in items1:
        if it in items2:
            res.append(it)
    if len(res) < 100:
        for it in items2:
            if it not in res:
                res.append(it)
    return res[:100]


def offline_eval(pred_val, data_dict):
    valid_run = data_dict['valid_run']
    valid_qrel = data_dict['valid_qrel']

    # 聚合itemId成list
    recom_df = pred_val.groupby(['userId'])['itemId'].agg(list).reset_index()
    recom_df.columns = ['userId', 'pred_itemIds']

    # 合并验证集itemIds
    recom_df = recom_df.merge(valid_run, on='userId', how='left')
    recom_df['itemIds'] = recom_df['itemIds'].apply(lambda x: x.split(','))

    recom_df['result_itemIds'] = recom_df.apply(
        lambda row: match_func(row['pred_itemIds'], row['itemIds']), axis=1)
    valid_recom_df = recom_df.merge(valid_qrel, on='userId', how='left')

    # 计算NDCG分数
    NDCG = 0
    for items in valid_recom_df[['result_itemIds', 'itemId']].values:
        l1 = items[0][:10]
        l2 = [items[1]]
        NDCG += getNDCG(l1, l2)
    NDCG = NDCG / len(valid_run)
    print('NDCG: ', NDCG)


def get_submit_file(test_file, pred):
    dict_df = dict(zip(pred['userId'] + '_' + pred['itemId'], pred['score']))
    submit_file = test_file.copy()
    submit_file['itemIds'] = submit_file['itemIds'].apply(
        lambda x: x.split(','))
    submit_file = submit_file.explode('itemIds')
    submit_file['key'] = submit_file['userId'] + '_' + submit_file['itemIds']
    submit_file['score'] = submit_file.apply(
        lambda x: dict_df.get(x['key'], 0.0), axis=1)
    submit_file = submit_file[['userId', 'itemIds', 'score']]
    submit_file.columns = ['userId', 'itemId', 'score']
    submit_file = submit_file.groupby('userId', group_keys=False).apply(
        lambda x: x.sort_values('score', ascending=False)).reset_index(
            drop=True)
    return submit_file


def get_all_data_without_val():
    data_dict_t1 = get_data_single(dirnm='t1')
    data_dict_t2 = get_data_single(dirnm='t2')
    data_dict_s1 = get_data_single(dirnm='s1')
    data_dict_s2 = get_data_single(dirnm='s2')
    data_dict_s3 = get_data_single(dirnm='s3')
    # with qrel:
    train_s1 = pd.concat([
        data_dict_s1['train'], data_dict_s1['train_5core'],
        data_dict_s1['valid_qrel']
    ],
                         ignore_index=True)
    train_s2 = pd.concat([
        data_dict_s2['train'], data_dict_s2['train_5core'],
        data_dict_s2['valid_qrel']
    ],
                         ignore_index=True)
    train_s3 = pd.concat([
        data_dict_s3['train'], data_dict_s3['train_5core'],
        data_dict_s3['valid_qrel']
    ],
                         ignore_index=True)
    train_t1 = pd.concat([data_dict_t1['train'], data_dict_t1['train_5core']],
                         ignore_index=True)
    train_t2 = pd.concat([data_dict_t2['train'], data_dict_t2['train_5core']],
                         ignore_index=True)
    tmp_train = pd.concat([train_s1, train_s2, train_s3, train_t1, train_t2],
                          ignore_index=True)
    tmp_train = tmp_train.drop_duplicates(
        subset=['userId', 'itemId']).reset_index(drop=True)
    return tmp_train


def train_main(target_market, t1t2_cross):
    tmp_train = get_all_data_without_val()
    if t1t2_cross:
        if target_market == 't1':
            tmp_train = pd.concat([tmp_train, DATA_dict['t2']['valid_qrel']],
                                  ignore_index=True)
        elif target_market == 't2':
            tmp_train = pd.concat([tmp_train, DATA_dict['t1']['valid_qrel']],
                                  ignore_index=True)
        else:
            raise
    model = llr_i2i(train_data=tmp_train)
    ## load item_sim_dict ...
    ## use pretrained item_sim_dict
    # model.item_sim_dict = item_sim_dict
    # model.built = True
    model.build_i2i_similarity_map()
    # Cache.cache_data(model.item_sim_dict, nm_marker=f"llr_i2i_all_data_cross__{target_market}")
    pred_val = model.predict(DATA_dict[target_market]['valid_run'])
    offline_eval(pred_val, data_dict=DATA_dict[target_market])
    pred_test = model.predict(DATA_dict[target_market]['test_run'])
    submit_val = get_submit_file(DATA_dict[target_market]['valid_run'],
                                 pred_val)
    submit_test = get_submit_file(DATA_dict[target_market]['test_run'],
                                  pred_test)
    submit_val.to_csv(f'{OUTPUT_PATH_dict[target_market]}/valid_pred.tsv',
                      sep="\t",
                      index=False)
    submit_test.to_csv(f'{OUTPUT_PATH_dict[target_market]}/test_pred.tsv',
                       sep="\t",
                       index=False)
    # NDCG:  0.7077831151664898
    # NDCG:  0.6548956968094127


if __name__ == '__main__':
    data_dict_t1 = get_data_single(dirnm='t1')
    data_dict_t2 = get_data_single(dirnm='t2')
    DATA_dict = {'t1': data_dict_t1, 't2': data_dict_t2}

    OUTPUT_DIRNM = "llr_2_train_only_t1t2_cross"
    OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR, OUTPUT_DIRNM)
    OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH, 't1')
    OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH, 't2')
    mkdirs(OUTPUT_PATH)
    mkdirs(OUTPUT_PATH_t1)
    mkdirs(OUTPUT_PATH_t2)
    OUTPUT_PATH_dict = {'t1': OUTPUT_PATH_t1, 't2': OUTPUT_PATH_t2}
    train_main(target_market='t1', t1t2_cross=True)
    train_main(target_market='t2', t1t2_cross=True)

    OUTPUT_DIRNM = "llr_2_train_only_no_t1t2_cross"
    OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR, OUTPUT_DIRNM)
    OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH, 't1')
    OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH, 't2')
    mkdirs(OUTPUT_PATH)
    mkdirs(OUTPUT_PATH_t1)
    mkdirs(OUTPUT_PATH_t2)
    OUTPUT_PATH_dict = {'t1': OUTPUT_PATH_t1, 't2': OUTPUT_PATH_t2}
    train_main(target_market='t1', t1t2_cross=False)
    train_main(target_market='t2', t1t2_cross=False)
