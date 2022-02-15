import sys

sys.path.insert(0, '/home/workspace')

# os.environ["CUDA_VISIBLE_DEVICES"]=""

import os
import gc
import math
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.botbase import mkdirs, bash2py, Cache
import pandas as pd
from src.xmrec_utils.io_utils import get_data_single
from src.xmrec_utils.config import OUTPUT_BASEDIR

OUTPUT_DIRNM = "bsl_v2"

OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR, "00_NEW", OUTPUT_DIRNM)
OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH, 't1')
OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH, 't2')
mkdirs(OUTPUT_PATH)
mkdirs(OUTPUT_PATH_t1)
mkdirs(OUTPUT_PATH_t2)


class itemcf_iuf:
    def __init__(self, train_data, norm_w=False):
        self.train_data = train_data
        self.norm_w = norm_w
        self.built = False

    def build_i2i_similarity_map(self):
        # self.train_data: # userId,itemId,rating
        user_item_ = self.train_data.groupby('userId')['itemId'].agg(
            list).reset_index()
        user_item_dict = dict(zip(user_item_['userId'], user_item_['itemId']))
        sim_item = {}  # {i0:{i1:score1,i2:score2,...}}}
        item_cnt = defaultdict(int)  # global count
        for user, items in tqdm(user_item_dict.items()):
            for item in items:
                item_cnt[item] += 1
                sim_item.setdefault(item, {})
                for relate_item in items:
                    if item == relate_item:
                        continue
                    sim_item[item].setdefault(relate_item, 0)
                    sim_item[item][relate_item] += 1 / math.log(
                        1 + len(items)
                    )  # longer is the sequence, more active is the user, lower weight is the item
        sim_item_corr = sim_item.copy()
        for i, related_items in tqdm(sim_item.items()):
            cur_max = -1.
            for j, cij in related_items.items():
                sim_item_corr[i][j] = cij / math.sqrt(
                    item_cnt[i] * item_cnt[j])
                cur_max = max(cur_max, sim_item_corr[i][j])
            cur_max += 1e-7
            if self.norm_w:
                for j, cij in related_items.items():
                    sim_item_corr[i][j] /= cur_max

        self.sim_item_corr = sim_item_corr
        self.user_item_dict = user_item_dict
        self.built = True

    def recommend(self, user_id):
        rank = defaultdict(int)
        try:
            interacted_items = self.user_item_dict[user_id]
        except:
            interacted_items = {}
        for i in interacted_items:
            try:
                for j, wij in sorted(self.sim_item_corr[i].items(),
                                     key=lambda d: d[1],
                                     reverse=True):  # 大->小
                    rank[j] += wij  # iterate and score item j
            except:
                pass
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)

    def predict(self, test_data):
        if not self.built:
            print("build i2i similarity map...")
            self.build_i2i_similarity_map()
        test_uid_pool = test_data['userId'].unique()
        recom_item = []
        for uid in tqdm(test_uid_pool):
            rank_item = self.recommend(uid)
            for j in rank_item:
                recom_item.append([uid, j[0], j[1]])


#                 if j[1] > 0.001:
#                     recom_item.append([uid, j[0], j[1]])
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


def add_score_train(train_file, pred):
    dict_df = dict(zip(pred['userId'] + '_' + pred['itemId'], pred['score']))
    submit_file = train_file.copy()
    submit_file['key'] = submit_file['userId'] + '_' + submit_file['itemId']
    submit_file['score'] = submit_file.apply(
        lambda x: dict_df.get(x['key'], 0.0), axis=1)
    submit_file = submit_file[['userId', 'itemId', 'score']]
    submit_file.columns = ['userId', 'itemId', 'score']
    return submit_file


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
train_t1 = pd.concat([
    data_dict_t1['train'], data_dict_t1['train_5core'],
    data_dict_t1['valid_qrel']
],
                     ignore_index=True)
train_t2 = pd.concat([
    data_dict_t2['train'], data_dict_t2['train_5core'],
    data_dict_t2['valid_qrel']
],
                     ignore_index=True)

# # without qrel:
# train_s1 = pd.concat([data_dict_s1['train'],data_dict_s1['train_5core'],data_dict_s1['valid_qrel']],ignore_index=True)
# train_s2 = pd.concat([data_dict_s2['train'],data_dict_s2['train_5core'],data_dict_s2['valid_qrel']],ignore_index=True)
# train_s3 = pd.concat([data_dict_s3['train'],data_dict_s3['train_5core'],data_dict_s3['valid_qrel']],ignore_index=True)
# train_t1 = pd.concat([data_dict_t1['train'],data_dict_t1['train_5core']],ignore_index=True)
# train_t2 = pd.concat([data_dict_t2['train'],data_dict_t2['train_5core']],ignore_index=True)
train_dt_all = pd.concat([train_s1, train_s2, train_s3, train_t1, train_t2],
                         ignore_index=True)

model = itemcf_iuf(train_dt_all, norm_w=False)
model.build_i2i_similarity_map()

pred_val = model.predict(data_dict_t1['valid_run'])
offline_eval(pred_val, data_dict=data_dict_t1)
pred_test = model.predict(data_dict_t1['test_run'])
# NDCG:  0.7773287230878129

Cache.dump_pkl(pred_test, f'{OUTPUT_PATH_t1}/test_pred_all.pkl')

submit_val = get_submit_file(data_dict_t1['valid_run'], pred_val)
submit_test = get_submit_file(data_dict_t1['test_run'], pred_test)

submit_val.to_csv(f'{OUTPUT_PATH_t1}/valid_pred.tsv', sep="\t", index=False)
submit_test.to_csv(f'{OUTPUT_PATH_t1}/test_pred.tsv', sep="\t", index=False)

# t2
# model = itemcf_iuf(pd.concat([train_dt_all, data_dict_t1['valid_qrel']],ignore_index=True))
pred_val = model.predict(data_dict_t2['valid_run'])
offline_eval(pred_val, data_dict=data_dict_t2)
pred_test = model.predict(data_dict_t2['test_run'])
# NDCG:  0.8419423153836637

Cache.dump_pkl(pred_test, f'{OUTPUT_PATH_t2}/test_pred_all.pkl')

submit_val = get_submit_file(data_dict_t2['valid_run'], pred_val)
submit_test = get_submit_file(data_dict_t2['test_run'], pred_test)
submit_val.to_csv(f'{OUTPUT_PATH_t2}/valid_pred.tsv', sep="\t", index=False)
submit_test.to_csv(f'{OUTPUT_PATH_t2}/test_pred.tsv', sep="\t", index=False)

bash2py(f"ls {OUTPUT_PATH}")
cmd = f"cd {OUTPUT_PATH} && zip -r submission.zip *"
bash2py(cmd)
bash2py(f"ls {OUTPUT_PATH}")