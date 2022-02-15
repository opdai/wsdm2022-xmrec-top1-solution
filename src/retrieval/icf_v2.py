import sys

sys.path.insert(0, '/home/workspace')
# os.environ["CUDA_VISIBLE_DEVICES"]=""
import os
import math
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from src.botbase import mkdirs, bash2py, Cache
from src.xmrec_utils.io_utils import get_data_single
from src.xmrec_utils.config import OUTPUT_BASEDIR, t1_test_run_path, t2_test_run_path


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
        for user, items in user_item_dict.items():
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
        for i, related_items in sim_item.items():
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
        for uid in test_uid_pool:
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


def get_all_data_by_markets(markets='s1-s2-s3-t1-t2',
                            add_tval=False,
                            cur_t='t1'):
    if cur_t not in ['t1', 't2']:
        raise
    all_corpus_lst = []
    markets = markets.split("-")
    for mkt in markets:
        data_dict = get_data_single(dirnm=mkt)
        data_dict['train_5core']['rating'] = 5.0
        all_corpus_lst += [data_dict['train'], data_dict['train_5core']]
        if (mkt == cur_t) and (not add_tval):
            continue
        all_corpus_lst += [data_dict['valid_qrel']]
    tmp_train = pd.concat(all_corpus_lst, axis=0, ignore_index=True)
    tmp_train = tmp_train.drop_duplicates(
        subset=['userId', 'itemId']).reset_index(drop=True)
    print(f"shape of {markets}: {tmp_train.shape}")
    return tmp_train


def train_and_infer(target_market, model=None):
    print(f"========== {target_market} ==========")
    if model is None:
        tmp_train = get_all_data_by_markets(markets=markets_map[target_market],
                                            add_tval=add_tval,
                                            cur_t=target_market)
        model = itemcf_iuf(tmp_train)
        model.build_i2i_similarity_map()
    pred_val = model.predict(data_dict[target_market]['valid_run'])
    offline_eval(pred_val, data_dict=data_dict[target_market])
    pred_test = model.predict(
        data_dict[target_market]['test_run'])  # NDCG:  0.7077831151664898
    # dump pred_test
    Cache.dump_pkl(pred_test,
                   f'{OUTPUT_PATH_dict[target_market]}/test_pred_all.pkl')

    submit_val = get_submit_file(data_dict[target_market]['valid_run'],
                                 pred_val)
    submit_test = get_submit_file(data_dict[target_market]['test_run'],
                                  pred_test)
    submit_val.to_csv(f'{OUTPUT_PATH_dict[target_market]}/valid_pred.tsv',
                      sep="\t",
                      index=False)
    submit_test.to_csv(f'{OUTPUT_PATH_dict[target_market]}/test_pred.tsv',
                       sep="\t",
                       index=False)

    # negsample_files = [f for f in os.listdir(os.path.join('/home/workspace/DATA/', target_market)) if
    #                     'negsample' in f]
    # for f in negsample_files:
    #     print(f"--------- Infer {f} -----------")
    #     # if 'run' in f:
    #     #     df_neg = pd.read_csv(os.path.join('/home/workspace/DATA/', target_market, f), sep='\t', header=None,names=['userId','itemIds'])
    #     # else:
    #     df_neg = pd.read_csv(os.path.join('/home/workspace/DATA/', target_market, f), sep='\t')
    #     pred = model.predict(df_neg)
    #     df_neg = df_neg.merge(pred, how='left', on=['userId', 'itemId'])
    #     output_name = f[:-4] if 'run' not in f else f[:-8]
    #     df_neg.to_csv(f'{OUTPUT_PATH_dict[target_market]}/{output_name}_pred.tsv', sep="\t", index=False)

    return model


def make_pred(test_run_path, mkt):
    print(f">>> Do inference for {mkt}: {test_run_path}")
    new_test_run = pd.read_csv(test_run_path,
                               sep='\t',
                               header=None,
                               names=['userId', 'itemIds'])
    # load score df
    test_pred_all = Cache.load_pkl(
        f'{OUTPUT_PATH_dict[mkt]}/test_pred_all.pkl')
    submit_test_new = get_submit_file(test_file=new_test_run,
                                      pred=test_pred_all)
    submit_test_new.to_csv(f'{OUTPUT_PATH_dict[mkt]}/test_pred_new.tsv',
                           sep="\t",
                           index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirnm', type=str, default='default')
    parser.add_argument('--sub_dirnm', type=str, default='00_NEW')
    parser.add_argument('--t1_markets', type=str, default='s1-s2-s3-t1-t2')
    parser.add_argument('--t2_markets', type=str, default='s1-s2-s3-t1-t2')
    parser.add_argument('--add_tval', type=int, default=0)
    parser.add_argument('--infer_only', type=int, default=0)
    # parser.add_argument('--t1_test_run_path', type=str, default='NONE_NONE')
    # parser.add_argument('--t2_test_run_path', type=str, default='NONE_NONE')

    ARGS = parser.parse_args()
    print("=====" * 20)
    for k, v in ARGS.__dict__.items():
        print(f">>> {k}: {v}")
    print("=====" * 20)

    OUTPUT_DIRNM = ARGS.output_dirnm
    SUB_DIR_NM = ARGS.sub_dirnm
    t1_markets = ARGS.t1_markets
    t2_markets = ARGS.t2_markets
    add_tval = ARGS.add_tval
    infer_only = ARGS.infer_only
    assert add_tval in [0, 1]
    assert infer_only in [0, 1]
    add_tval = bool(add_tval)
    infer_only = bool(infer_only)

    markets_map = {'t1': t1_markets, 't2': t2_markets}
    # t1_test_run_path = ARGS.t1_test_run_path
    # t2_test_run_path = ARGS.t2_test_run_path

    # # t1_test_run_path = '/home/workspace/DATA/t1/test_run.tsv'
    # # t2_test_run_path = '/home/workspace/DATA/t2/test_run.tsv'

    OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR, SUB_DIR_NM, OUTPUT_DIRNM)
    OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH, 't1')
    OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH, 't2')
    mkdirs(OUTPUT_PATH)
    mkdirs(OUTPUT_PATH_t1)
    mkdirs(OUTPUT_PATH_t2)
    OUTPUT_PATH_dict = {'t1': OUTPUT_PATH_t1, 't2': OUTPUT_PATH_t2}
    data_dict_t1 = get_data_single(dirnm='t1')
    data_dict_t2 = get_data_single(dirnm='t2')
    data_dict = {'t1': data_dict_t1, 't2': data_dict_t2}

    print(f"t1_markets: {t1_markets}")
    print(f"t2_markets: {t2_markets}")
    print(f"add_tval: {add_tval}")

    if not infer_only:
        model = train_and_infer(target_market='t1', model=None)
        if (t1_markets != t2_markets) or ((('t1' in t1_markets) or
                                           ('t2' in t1_markets)) and
                                          (not add_tval)):
            print("train new model for t2 ...")
            model = train_and_infer(target_market='t2', model=None)
        else:
            print("use t1 model for t2 ...")
            model = train_and_infer(target_market='t2', model=model)
        # bash2py(f"ls {OUTPUT_PATH}")
        # cmd = f"cd {OUTPUT_PATH} && zip -r submission.zip *"
        # bash2py(cmd)
        # bash2py(f"ls {OUTPUT_PATH}")

    if t1_test_run_path != 'NONE_NONE':
        make_pred(test_run_path=t1_test_run_path, mkt='t1')

    if t2_test_run_path != 'NONE_NONE':
        make_pred(test_run_path=t2_test_run_path, mkt='t2')