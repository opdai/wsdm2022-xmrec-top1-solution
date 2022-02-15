import sys

sys.path.insert(0, '/home/workspace')
# os.environ["CUDA_VISIBLE_DEVICES"]=""
import os
import math
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from src.botbase import mkdirs, bash2py
from src.xmrec_utils.io_utils import get_data_single
from src.xmrec_utils.config import OUTPUT_BASEDIR
import pickle
from tqdm import tqdm


class usercf:
    def __init__(self, train_data, ucf_note='00_', norm_w=False):
        self.train_data = train_data
        self.norm_w = norm_w
        self.built = False
        self.ucf_note = ucf_note

    def build_u2u_similarity_map(self, topk=0, save=False):
        # self.train_data: # userId,itemId,rating
        item_user_ = self.train_data.groupby('itemId')['userId'].agg(list).reset_index()
        item_user_dict = dict(zip(item_user_['itemId'], item_user_['userId']))
        sim_user = {}  # {u0:{u1:score1,u2:score2,...}}}
        user_cnt = defaultdict(int)  # global count
        for item, users in item_user_dict.items():
            for user in users:
                user_cnt[user] += 1
                sim_user.setdefault(user, {})
                for relate_user in users:
                    if user == relate_user:
                        continue
                    sim_user[user].setdefault(relate_user, 0)
                    # 对热门item进行惩罚 [User-IIF] (/ math.log(1 + len(users)))
                    sim_user[user][relate_user] += 1 / math.log(1 + len(users))
        sim_user_corr = sim_user.copy()
        for i, related_users in sim_user.items():
            cur_max = -1.
            for j, cij in related_users.items():
                sim_user_corr[i][j] = cij / math.sqrt(user_cnt[i] * user_cnt[j])
                cur_max = max(cur_max, sim_user_corr[i][j])
            cur_max += 1e-7
            if self.norm_w:
                for j, cij in related_users.items():
                    sim_user_corr[i][j] /= cur_max
        del sim_user

        sim_user = sim_user_corr.copy()
        for user, related_users in sim_user_corr.items():
            if 's' in user:
                sim_user.pop(user)
                continue
            sorted_users = sorted(related_users.items(), key=lambda d: d[1], reverse=True)
            if topk and len(sorted_users) > topk:
                sorted_users = sorted_users[:topk]
            sim_user[user] = dict(sorted_users)

        self.sim_user_corr = sim_user
        self.item_user_dict = item_user_dict
        self.built = True

        if save:
            with open(os.path.join(OUTPUT_PATH, f'{self.ucf_note}sim_user.pkl'), 'wb') as f:
                pickle.dump(sim_user, f)
            # with open(os.path.join(OUTPUT_PATH, f'{self.ucf_note}item_user_dict.pkl'), 'wb') as f:
            #     pickle.dump(item_user_dict, f)

    def recommend(self, user_id):
        user_item_ = self.train_data.groupby('userId')['itemId'].agg(list).reset_index()
        user_item_dict = dict(zip(user_item_['userId'], user_item_['itemId']))

        rank = defaultdict(int)

        try:
            related_users = self.sim_user_corr[user_id]
        except:
            return rank

        for related_user, u2u_sim in related_users.items():
            try:
                interacted_items = user_item_dict[related_user]
            except:
                interacted_items = {}
            for i in interacted_items:
                rank[i] += u2u_sim
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)

    def predict(self, test_data):
        if not self.built:
            try:
                print(f'load u2u similarity map from f{OUTPATH}')
                with open(os.path.join(OUTPUT_PATH, f'{self.ucf_note}sim_user.pkl'), 'rb') as f:
                    self.sim_user_corr = pickle.load(f)
                # with open(os.path.join(OUTPUT_PATH, f'{self.ucf_note}item_user_dict.pkl'), 'rb') as f:
                #     self.item_user_dict = pickle.load(f)
            except:
                print("build u2u similarity map...")
                self.build_u2u_similarity_map()
        test_uid_pool = test_data['userId'].unique()
        recom_item = []
        for uid in tqdm(test_uid_pool):
            rank_item = self.recommend(uid)
            if rank_item:
                for j in rank_item:
                    recom_item.append([uid, j[0], j[1]])
            #                 if j[1] > 0.001:
            #                     recom_item.append([uid, j[0], j[1]])
        recom_item_df = pd.DataFrame(recom_item)
        recom_item_df.columns = ['userId', 'itemId', 'score']
        return recom_item_df

    def predict_new(self, test_data):
        # 只计算test_data中的 u-i pair对应的分数
        if not self.built:
            try:
                print(f'load u2u similarity map from {OUTPUT_PATH}')
                with open(os.path.join(OUTPUT_PATH, f'{self.ucf_note}sim_user.pkl'), 'rb') as f:
                    self.sim_user_corr = pickle.load(f)
                    self.built = True
                # with open(os.path.join(OUTPUT_PATH, f'{self.ucf_note}_item_user_dict.pkl'), 'rb') as f:
                #     self.item_user_dict = pickle.load(f)
            except:
                print("build u2u similarity map...")
                self.build_u2u_similarity_map()

        test_data_copy = test_data.copy()
        if 'itemId' in test_data.columns:
            # userId-itemId 格式的数据
            test_data_copy = test_data_copy.groupby('userId')['itemId'].agg(list).reset_index()
            test_data_copy = test_data_copy.rename(columns={'itemId': 'itemIds'})
        else:
            test_data_copy.itemIds = test_data_copy.itemIds.apply(lambda x: x.split(','))
        test_user_item_dict = dict(zip(test_data_copy['userId'], test_data_copy['itemIds']))
        # train-data
        user_item_ = self.train_data.groupby('userId')['itemId'].agg(list).reset_index()
        user_item_dict = dict(zip(user_item_['userId'], user_item_['itemId']))

        recom_item = []
        for user, items in tqdm(test_user_item_dict.items(), total=len(test_user_item_dict)):
            rank = defaultdict(int)
            related_users = self.sim_user_corr.get(user, {})
            for item in items:
                for related_user, score in related_users.items():
                    if item in user_item_dict[related_user]:
                        rank[item] += score
                recom_item.append([user, item, rank.get(item, 0)])
        recom_item_df = pd.DataFrame(recom_item)
        recom_item_df.columns = ['userId', 'itemId', 'score']
        return recom_item_df

    def predict_multiprocessing(self, test_data):
        from multiprocessing import Pool as ProcessPool
        if not self.built:
            print("build i2i similarity map...")
            self.build_u2u_similarity_map()
        test_uid_pool = test_data['userId'].unique()
        print(len(test_uid_pool))

        pool = ProcessPool(NUM_PROCESS)

        #         res_list = pool.map(self.recommend, test_uid_pool)
        res_list = list(tqdm(pool.imap(self.recommend, test_uid_pool), total=len(test_uid_pool)))
        pool.close()
        pool.join()

        recom_item = []
        for uid, rank_item in zip(test_uid_pool, res_list):
            if rank_item:
                for j in rank_item:
                    recom_item.append([uid, j[0], j[1]])

        recom_item_df = pd.DataFrame(recom_item)
        recom_item_df.columns = ['userId', 'itemId', 'score']
        return recom_item_df


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)  # 生成IDCG的rankscore
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
    pred_val = pred_val.sort_values(by='score', ascending=False)
    valid_run = data_dict['valid_run']
    valid_qrel = data_dict['valid_qrel']

    # 聚合itemId成list
    recom_df = pred_val.groupby(['userId'])['itemId'].agg(list).reset_index()
    recom_df.columns = ['userId', 'pred_itemIds']

    # 合并验证集itemIds
    recom_df = recom_df.merge(valid_run, on='userId', how='left')
    recom_df['itemIds'] = recom_df['itemIds'].apply(lambda x: x.split(','))

    recom_df['result_itemIds'] = recom_df.apply(lambda row: match_func(row['pred_itemIds'], row['itemIds']), axis=1)
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
    submit_file['itemIds'] = submit_file['itemIds'].apply(lambda x: x.split(','))
    submit_file = submit_file.explode('itemIds')
    submit_file['key'] = submit_file['userId'] + '_' + submit_file['itemIds']
    submit_file['score'] = submit_file.apply(lambda x: dict_df.get(x['key'], 0.0), axis=1)
    submit_file = submit_file[['userId', 'itemIds', 'score']]
    submit_file.columns = ['userId', 'itemId', 'score']
    submit_file = submit_file.groupby('userId', group_keys=False).apply(
        lambda x: x.sort_values('score', ascending=False)).reset_index(drop=True)
    return submit_file


def get_all_data_by_markets(markets='s1-s2-s3-t1-t2', add_tval=False, cur_t='t1'):
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
    tmp_train = tmp_train.drop_duplicates(subset=['userId', 'itemId']).reset_index(drop=True)
    print(f"shape of {markets}: {tmp_train.shape}")
    return tmp_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirnm', type=str, default='default')
    parser.add_argument('--sub_dirnm', type=str, default='00_NEW')
    parser.add_argument('--t1_markets', type=str, default='s1-s2-s3-t1-t2')
    parser.add_argument('--t2_markets', type=str, default='s1-s2-s3-t1-t2')
    parser.add_argument('--add_tval', type=int, default=0)
    parser.add_argument('--infer_neg', type=int, default=0)

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
    assert add_tval in [0, 1]
    add_tval = bool(add_tval)
    infer_neg = bool(ARGS.infer_neg)

    markets_map = {'t1': t1_markets, 't2': t2_markets}
    OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR, SUB_DIR_NM, OUTPUT_DIRNM)
    OUTPUT_PATH_s1 = os.path.join(OUTPUT_PATH, 's1')
    OUTPUT_PATH_s2 = os.path.join(OUTPUT_PATH, 's2')
    OUTPUT_PATH_s3 = os.path.join(OUTPUT_PATH, 's3')
    OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH, 't1')
    OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH, 't2')
    mkdirs(OUTPUT_PATH)
    mkdirs(OUTPUT_PATH_t1)
    mkdirs(OUTPUT_PATH_t2)
    mkdirs(OUTPUT_PATH_s1)
    mkdirs(OUTPUT_PATH_s2)
    mkdirs(OUTPUT_PATH_s3)
    OUTPUT_PATH_dict = {'t1': OUTPUT_PATH_t1, 't2': OUTPUT_PATH_t2, 's1':OUTPUT_PATH_s1, 's2':OUTPUT_PATH_s2, 's3':OUTPUT_PATH_s3}
    data_dict_t1 = get_data_single(dirnm='t1')
    data_dict_t2 = get_data_single(dirnm='t2')
    data_dict = {'t1': data_dict_t1, 't2': data_dict_t2}


    def train_and_infer(target_market, model=None):
        print(f"========== {target_market} ==========")
        if model is None:
            tmp_train = get_all_data_by_markets(markets=markets_map[target_market], add_tval=add_tval,
                                                cur_t=target_market)
            model = usercf(tmp_train, ucf_note=target_market)
            model.build_u2u_similarity_map(save=True)
        pred_val = model.predict_new(data_dict[target_market]['valid_run'])
        offline_eval(pred_val, data_dict=data_dict[target_market])
        pred_test = model.predict_new(data_dict[target_market]['test_run'])  # NDCG:  0.7077831151664898
        submit_val = get_submit_file(data_dict[target_market]['valid_run'], pred_val)
        submit_test = get_submit_file(data_dict[target_market]['test_run'], pred_test)
        submit_val.to_csv(f'{OUTPUT_PATH_dict[target_market]}/valid_pred.tsv', sep="\t", index=False)
        submit_test.to_csv(f'{OUTPUT_PATH_dict[target_market]}/test_pred.tsv', sep="\t", index=False)
        return model


    def infer_negsample(target_market):
        print(f"========== {target_market} ==========")
        tmp_train = get_all_data_by_markets(markets=markets_map[target_market], add_tval=add_tval,
                                            cur_t=target_market)
        model = usercf(tmp_train, ucf_note=target_market)

        # note: 2del 'run'
        negsample_files = [f for f in os.listdir(os.path.join('/home/workspace/DATA/', target_market)) if 'negsample' in f]
        for f in negsample_files:
            print(f"--------- Infer {f} -----------")
            # if 'run' in f:
            #     df_neg = pd.read_csv(os.path.join('/home/workspace/DATA/', target_market, f), sep='\t', header=None,names=['userId','itemIds'])
            # else:
            df_neg = pd.read_csv(os.path.join('/home/workspace/DATA/', target_market, f), sep='\t')
            pred = model.predict_new(df_neg)
            assert len(df_neg) == len(pred)
            output_name = f[:-4] if 'run' not in f else f[:-8]
            pred.to_csv(f'{OUTPUT_PATH_dict[target_market]}/{output_name}_pred.tsv', sep="\t", index=False)

    if not infer_neg:
        model = train_and_infer(target_market='t1', model=None)
        print(f"t1_markets: {t1_markets}")
        print(f"t2_markets: {t2_markets}")
        print(f"add_tval: {add_tval}")
        if (t1_markets != t2_markets) or ((('t1' in t1_markets) or ('t2' in t1_markets)) and (not add_tval)):
            print("train new model for t2 ...")
            model = train_and_infer(target_market='t2', model=None)
        else:
            print("use t1 model for t2 ...")
            model = train_and_infer(target_market='t2', model=model)
    else:
        print("Infer negative samples...")
        for tgt in ['t1', 't2']:
            infer_negsample(tgt)

    # bash2py(f"ls {OUTPUT_PATH}")
    # cmd = f"cd {OUTPUT_PATH} && zip -r submission.zip *"
    # bash2py(cmd)
    # bash2py(f"ls {OUTPUT_PATH}")