import sys
sys.path.insert(0, '/home/workspace')
# os.environ["CUDA_VISIBLE_DEVICES"]=""
import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from src.botbase import mkdirs,Cache
from src.xmrec_utils.io_utils import get_data_single
from itertools import combinations,permutations
from src.xmrec_utils.config import OUTPUT_BASEDIR,t1_test_run_path,t2_test_run_path

class swing:
    def __init__(self, train_data,alpha = 0.01):
        self.train_data = train_data
        self.alpha = alpha
        self.u2i_dict = defaultdict(set)
        self.i2u_dict = defaultdict(set)
        for _, row in self.train_data.iterrows():
            self.u2i_dict[row["userId"]].add(row["itemId"])
            self.i2u_dict[row["itemId"]].add(row["userId"])
        
    def build_i2i_similarity_map(self):
        # item_pairs = list(permutations(self.i2u_dict.keys(), 2))
        item_pairs_set = set()
        # user_pairs_set = set()
        for u, items in self.u2i_dict.items():
            item_pairs_set.update(permutations(items, 2))
        # for i, users in self.i2u_dict.items():
        #     user_pairs_set.update(combinations(users, 2))
        # user_pairs = list(user_pairs_set)
        print(len(item_pairs_set))
        # print(len(user_pairs))
        # 补充可能被忽略的u-i-u-i关系中的 i-i pair
        # for (u1, u2) in tqdm(user_pairs):
        #     u1_items = self.u2i_dict[u1]
        #     u2_items = self.u2i_dict[u2]
        #     item_intersection = u1_items.intersection(u2_items)
        #     if len(item_intersection) > 1:
        #         item_pairs_set.update(combinations(item_intersection, 2))

        item_pairs = list(item_pairs_set)

        print(f"item pairs length: {len(item_pairs)}")
        item_sim_dict = defaultdict(dict)
        for (i, j) in item_pairs:
            if i in item_sim_dict[j] and isinstance(item_sim_dict[j][i],(int,float,np.number)):
                item_sim_dict[i][j] = item_sim_dict[j][i]
                continue
            user_pairs = list(permutations(self.i2u_dict[i] & self.i2u_dict[j], 2))
            result = 0.0
            for (u, v) in user_pairs:
                result += 1 / (self.alpha + len(self.u2i_dict[u] & self.u2i_dict[v]))
            item_sim_dict[i][j] = result
        self.item_sim_dict = item_sim_dict
        self.built = True
        # with open('i2i_dic_new.pkl', 'wb') as f:
        #     pickle.dump(self.item_sim_dict, f)

            
    def recommend(self,user_id):  
        rank = defaultdict(int)
        try:
            interacted_items = self.u2i_dict[user_id]
        except:
            interacted_items = {}  
        for i in interacted_items: # 历史交互列表
            try:
                for j, wij in sorted(self.item_sim_dict[i].items(),
                                     key=lambda d: d[1], reverse=True):  # 大->小
                    rank[j] += wij
            except:
                pass
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)

    def predict(self,test_data):
        if not self.built:
            print("build i2i similarity map...")
            self.build_i2i_similarity_map()
        test_uid_pool = test_data['userId'].unique()
        # 进行召回
        recom_item = []
        for uid in test_uid_pool:
            rank_item = self.recommend(uid)
            for j in rank_item:  
                recom_item.append([uid, j[0], j[1]])
        recom_item_df = pd.DataFrame(recom_item)
        recom_item_df.columns = ['userId','itemId','score']
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

def offline_eval(pred_val,data_dict):
    pred_val = pred_val.sort_values(by='score', ascending=False)
    valid_run=data_dict['valid_run']
    valid_qrel=data_dict['valid_qrel']

    # 聚合itemId成list
    recom_df = pred_val.groupby(['userId'])['itemId'].agg(list).reset_index()
    recom_df.columns = ['userId','pred_itemIds']

    # 合并验证集itemIds
    recom_df = recom_df.merge(valid_run, on='userId', how='left')
    recom_df['itemIds'] =  recom_df['itemIds'].apply(lambda x:x.split(','))

    recom_df['result_itemIds'] = recom_df.apply(lambda row: match_func(row['pred_itemIds'], row['itemIds']), axis = 1)
    valid_recom_df = recom_df.merge(valid_qrel, on='userId', how='left')

    # 计算NDCG分数
    NDCG = 0
    for items in valid_recom_df[['result_itemIds','itemId']].values:
        l1 = items[0][:10]
        l2 = [items[1]]
        NDCG += getNDCG(l1, l2)
    NDCG = NDCG/len(valid_run)
    print('NDCG: ', NDCG)


def get_submit_file(test_file,pred):
    dict_df = dict(zip(pred['userId'] + '_' + pred['itemId'], pred['score'])) 
    submit_file = test_file.copy()
    submit_file['itemIds'] =  submit_file['itemIds'].apply(lambda x:x.split(','))
    submit_file = submit_file.explode('itemIds')
    submit_file['key'] = submit_file['userId'] + '_' + submit_file['itemIds']
    submit_file['score'] = submit_file.apply(lambda x:dict_df.get(x['key'],0.0), axis=1)
    submit_file = submit_file[['userId','itemIds','score']]
    submit_file.columns = ['userId','itemId','score']
    submit_file = submit_file.groupby('userId', group_keys=False).apply(lambda x: x.sort_values('score', ascending=False)).reset_index(drop=True)
    return submit_file


def get_all_data_by_markets(markets='s1-s2-s3-t1-t2',add_tval=False, add_crossval=False, cur_t='t1'):
    if cur_t not in ['t1','t2']:
        raise
    all_corpus_lst = []
    markets = markets.split("-")
    for mkt in markets:
        data_dict = get_data_single(dirnm=mkt)
        data_dict['train_5core']['rating'] = 5.0
        all_corpus_lst+=[data_dict['train'],data_dict['train_5core']]
        if (mkt == cur_t) and (not add_tval):
            continue
        if (not add_crossval) and (mkt in ['t1', 't2']):
            continue
        all_corpus_lst+=[data_dict['valid_qrel']]
    tmp_train = pd.concat(all_corpus_lst,axis=0,ignore_index=True)
    tmp_train = tmp_train.drop_duplicates(subset=['userId','itemId']).reset_index(drop=True)
    print(f"shape of {markets}: {tmp_train.shape}")
    return tmp_train


def train_and_infer(target_market, model=None):
    print(f"========== {target_market} ==========")
    if model is None:
        tmp_train = get_all_data_by_markets(markets=markets_map[target_market], add_tval=add_tval, add_crossval=add_crossval, cur_t=target_market)
        model = swing(tmp_train)
        model.build_i2i_similarity_map()
    pred_val = model.predict(data_dict[target_market]['valid_run'])
    offline_eval(pred_val, data_dict=data_dict[target_market])
    pred_test = model.predict(data_dict[target_market]['test_run']) # NDCG:  0.7077831151664898
    # dump pred_test
    Cache.dump_pkl(pred_test, f'{OUTPUT_PATH_dict[target_market]}/test_pred_all.pkl')

    submit_val = get_submit_file(data_dict[target_market]['valid_run'],pred_val)
    submit_test = get_submit_file(data_dict[target_market]['test_run'],pred_test)
    submit_val.to_csv(f'{OUTPUT_PATH_dict[target_market]}/valid_pred.tsv', sep="\t", index=False)
    submit_test.to_csv(f'{OUTPUT_PATH_dict[target_market]}/test_pred.tsv', sep="\t", index=False)
    
    # negsample_files = [f for f in os.listdir(os.path.join('/home/workspace/DATA/', target_market)) if
    #                    'negsample' in f]
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
    new_test_run = pd.read_csv(test_run_path, sep='\t', header=None, names=['userId','itemIds'])
    # load score df
    test_pred_all = Cache.load_pkl(f'{OUTPUT_PATH_dict[mkt]}/test_pred_all.pkl')
    submit_test_new = get_submit_file(test_file= new_test_run, pred=test_pred_all)
    submit_test_new.to_csv(f'{OUTPUT_PATH_dict[mkt]}/test_pred_new.tsv', sep="\t", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirnm', type=str, default='default')
    parser.add_argument('--sub_dirnm', type=str, default='00_NEW')
    parser.add_argument('--t1_markets', type=str, default='s1-s2-s3-t1-t2')
    parser.add_argument('--t2_markets', type=str, default='s1-s2-s3-t1-t2')
    parser.add_argument('--add_tval', type=int, default=0)
    parser.add_argument('--add_crossval', type=int, default=1)
    parser.add_argument('--infer_only', type=int, default=0)
    # parser.add_argument('--t1_test_run_path', type=str, default='NONE_NONE')
    # parser.add_argument('--t2_test_run_path', type=str, default='NONE_NONE')

    ARGS = parser.parse_args()
    print("====="*20)
    for k,v in ARGS.__dict__.items():
        print(f">>> {k}: {v}")
    print("====="*20)

    OUTPUT_DIRNM=ARGS.output_dirnm
    SUB_DIR_NM = ARGS.sub_dirnm
    t1_markets=ARGS.t1_markets
    t2_markets=ARGS.t2_markets
    add_tval = ARGS.add_tval
    add_crossval = ARGS.add_crossval
    infer_only = ARGS.infer_only
    assert add_tval in [0, 1]
    assert add_crossval in [0, 1]
    assert infer_only in [0, 1]
    add_tval = bool(add_tval)
    add_crossval = bool(add_crossval)
    infer_only = bool(infer_only)
    
    markets_map = {'t1':t1_markets, 't2':t2_markets}
    # t1_test_run_path = ARGS.t1_test_run_path
    # t2_test_run_path = ARGS.t2_test_run_path

    # # t1_test_run_path = '/home/workspace/DATA/t1/test_run.tsv'
    # # t2_test_run_path = '/home/workspace/DATA/t2/test_run.tsv'

    OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR,SUB_DIR_NM,OUTPUT_DIRNM)
    OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH,'t1')
    OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH,'t2')
    mkdirs(OUTPUT_PATH)
    mkdirs(OUTPUT_PATH_t1)
    mkdirs(OUTPUT_PATH_t2)
    OUTPUT_PATH_dict = {'t1':OUTPUT_PATH_t1, 't2':OUTPUT_PATH_t2}
    data_dict_t1 = get_data_single(dirnm='t1')
    data_dict_t2 = get_data_single(dirnm='t2')
    data_dict = {'t1':data_dict_t1, 't2':data_dict_t2}


    print(f"t1_markets: {t1_markets}")
    print(f"t2_markets: {t2_markets}")
    print(f"add_tval: {add_tval}")

    if not infer_only:

        model = train_and_infer(target_market='t1', model=None)

        if (t1_markets != t2_markets) or ((('t1' in t1_markets) or ('t2' in t1_markets)) and (not add_tval) and add_crossval): # bug?
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