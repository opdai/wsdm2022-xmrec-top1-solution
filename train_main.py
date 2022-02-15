from email.policy import default
import sys

sys.path.insert(0, '/home/workspace')
import gc
import os
import argparse
import swifter
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from src.xmrec_utils.config import OUTPUT_BASEDIR
from src.xmrec_utils.io_utils import get_data_single
from src.botbase import Cache, mkdirs, bash2py
from src.botbase.utils import reduce_mem_usage
from src.xmrec_utils.evaluation import getNDCG
from src.feas_conf import T1_FEATURES_NEW_ALL, T2_FEATURES_NEW_ALL
import src.config as conf_all


def preprocess(valid_run, valid_qrel, test_run):
    val_data = valid_run.copy()
    test_data = test_run.copy()
    val_data['itemIds'] = val_data['itemIds'].apply(lambda x: x.split(','))
    val_data = val_data.explode('itemIds')
    val_data.columns = ['userId', 'itemId']
    val_data = val_data.merge(valid_qrel, how='left', on=['userId', 'itemId'])
    val_data['rating'] = val_data['rating'].fillna(0)
    val_data = val_data[['userId', 'itemId', 'rating']]
    val_data.columns = ['userId', 'itemId', 'label']
    test_data['itemIds'] = test_data['itemIds'].apply(lambda x: x.split(','))
    test_data = test_data.explode('itemIds')
    test_data.columns = ['userId', 'itemId']
    test_data = test_data[['userId', 'itemId']]
    return val_data, test_data


def get_data_df_all(target_market, new_test_run_path=None):
    data_dict = get_data_single(target_market)
    train_5core = data_dict['train_5core']
    train = data_dict['train']
    valid_qrel = data_dict['valid_qrel']
    valid_run = data_dict['valid_run']
    if (new_test_run_path is None) or (new_test_run_path == 'NONE_NONE'):
        print("Use old test_run.tsv !!!")
        test_run = data_dict['test_run']
    else:
        print(f"Use new test_run.tsv !!! {new_test_run_path}")
        test_run = pd.read_csv(new_test_run_path,
                               sep='\t',
                               header=None,
                               names=['userId', 'itemIds'])
    valid_df, test_df = preprocess(valid_run=valid_run,
                                   valid_qrel=valid_qrel,
                                   test_run=test_run)
    data_df = pd.concat([valid_df, test_df], axis=0, ignore_index=True)
    return data_df, train, train_5core


def ui_stats(df):
    uidf = df.groupby("userId").agg(UI_count=("itemId", "count"),
                                    UI_nunique=("itemId",
                                                "nunique")).reset_index()

    iudf = df.groupby("itemId").agg(IU_count=("userId", "count"),
                                    IU_nunique=("userId",
                                                "nunique")).reset_index()
    return uidf, iudf


def gen_w2v_feas(df,
                 sent_id,
                 word_id,
                 emb_size=conf_all.EMB_SIZE,
                 epochs=conf_all.EPOCHS,
                 window=conf_all.WINDOW,
                 min_count=conf_all.MIN_COUNT,
                 sg=conf_all.SG,
                 hs=conf_all.HS,
                 negative=conf_all.NEGATIVE,
                 n_jobs=conf_all.N_JOBS,
                 seed=conf_all.SEED,
                 nm_marker=''):
    tmp_col_nms = f'{sent_id}_{word_id}_list'
    tmp = df.groupby(sent_id, as_index=False)[word_id].agg({tmp_col_nms: list})
    sentences = tmp[tmp_col_nms].values.tolist()
    del tmp[tmp_col_nms]
    model = Word2Vec(
        sentences,
        vector_size=emb_size,
        window=window,
        workers=n_jobs,
        min_count=min_count,  # min_count>1 => OOV
        sg=sg,  # 1 for skip-gram; otherwise CBOW.
        hs=hs,  # If 1, hierarchical softmax will be used for model training
        negative=negative,  # hs=1 + negative
        epochs=epochs,
        seed=seed)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0.0] * emb_size)
    col_nms = [
        f'{nm_marker}__{sent_id}_{word_id}_emb_{ii}' for ii in range(emb_size)
    ]
    df_emb = pd.DataFrame(emb_matrix, columns=col_nms)
    return pd.concat([tmp, df_emb], axis=1), model


def get_wv(model_wv, key, emb_dim):
    try:
        return model_wv[key]
    except:
        return [np.nan] * emb_dim


def explode_emb(df, emb_column, emb_dim):
    for i in range(emb_dim):
        df[f'w2v_word_embedding_{i}'] = df[emb_column].apply(lambda x: x[i])
    return df


def func_calc_sim_base(row, model, vecs_all):
    try:
        target_item = row['itemId']
        item_lst = row['itemIds']
        target_vec = vecs_all[model.wv.key_to_index[target_item]]
        simi_lst = []
        for item in item_lst:
            cur_vec = vecs_all[model.wv.key_to_index[item]]
            cur_simi = np.dot(target_vec, cur_vec)
            simi_lst.append(cur_simi)
        return [
            np.sum(simi_lst),
            np.max(simi_lst),
            np.min(simi_lst),
            np.std(simi_lst),
            np.mean(simi_lst),
            np.median(simi_lst),
            np.percentile(simi_lst, 25),
            np.percentile(simi_lst, 75),
            np.percentile(simi_lst, 5),
            np.percentile(simi_lst, 95)
        ]
    except:
        return [
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan
        ]


EMBDIM = 32


def gen_feas_all(target_market, new_test_run_path):

    df_raw_tx, train_tx, train_5core_tx = get_data_df_all(
        target_market, new_test_run_path=new_test_run_path)

    # step 1
    uidf_train, iudf_train = ui_stats(df=train_tx)
    df_raw_tx = df_raw_tx.merge(uidf_train, on='userId', how='left')
    df_raw_tx = df_raw_tx.merge(iudf_train, on='itemId', how='left')
    uidf_train, iudf_train = ui_stats(df=train_5core_tx)
    uidf_train = uidf_train.rename(columns={
        'UI_count': 'UI_count_5core',
        'UI_nunique': 'UI_nunique_5core'
    })
    iudf_train = iudf_train.rename(columns={
        'IU_count': 'IU_count_5core',
        'IU_nunique': 'IU_nunique_5core'
    })
    df_raw_tx = df_raw_tx.merge(uidf_train, on='userId', how='left')
    df_raw_tx = df_raw_tx.merge(iudf_train, on='itemId', how='left')

    # step 2
    tmp_u = train_tx.groupby('userId')['rating'].mean().reset_index()
    tmp_u = tmp_u.rename(columns={'rating': 'user_rating_mean'})
    tmp_i = train_tx.groupby('itemId')['rating'].mean().reset_index()
    tmp_i = tmp_i.rename(columns={'rating': 'item_rating_mean'})
    df_raw_tx = df_raw_tx.merge(tmp_u, on='userId', how='left')
    df_raw_tx = df_raw_tx.merge(tmp_i, on='itemId', how='left')

    # step 3
    emb_01, model_01 = gen_w2v_feas(df=train_tx,
                                    sent_id='userId',
                                    word_id='itemId',
                                    emb_size=EMBDIM,
                                    nm_marker='train')
    model_01_vecs_all = model_01.wv.vectors / np.linalg.norm(
        model_01.wv.vectors, axis=1, keepdims=True)
    func_calc_sim = partial(func_calc_sim_base,
                            model=model_01,
                            vecs_all=model_01_vecs_all)

    emb_02, _ = gen_w2v_feas(df=train_tx,
                             sent_id='itemId',
                             word_id='userId',
                             emb_size=EMBDIM,
                             nm_marker='train')

    emb_03, _ = gen_w2v_feas(df=train_5core_tx,
                             sent_id='userId',
                             word_id='itemId',
                             emb_size=EMBDIM,
                             nm_marker='train_5core')

    emb_04, _ = gen_w2v_feas(df=train_5core_tx,
                             sent_id='itemId',
                             word_id='userId',
                             emb_size=EMBDIM,
                             nm_marker='train_5core')

    df_raw_tx = df_raw_tx.merge(emb_01, on='userId', how='left')
    df_raw_tx = df_raw_tx.merge(emb_02, on='itemId', how='left')
    df_raw_tx = df_raw_tx.merge(emb_03, on='userId', how='left')
    df_raw_tx = df_raw_tx.merge(emb_04, on='itemId', how='left')

    # step 4
    # simi feas creates
    train_tx_user_item_list = train_tx.groupby('userId')['itemId'].agg(
        list).reset_index()
    train_tx_user_item_list = train_tx_user_item_list.rename(
        columns={'itemId': 'itemIds'})

    tmp = df_raw_tx[['userId',
                     'itemId']].drop_duplicates().reset_index(drop=True)
    tmp = tmp.merge(train_tx_user_item_list, on='userId', how='left')
    tmp['simis'] = tmp[['itemIds',
                        'itemId']].swifter.set_npartitions(32).apply(
                            lambda x: func_calc_sim(x), axis=1)

    tmp['simi_scores_sum'] = tmp['simis'].apply(lambda x: x[0])
    tmp['simi_scores_max'] = tmp['simis'].apply(lambda x: x[1])
    tmp['simi_scores_min'] = tmp['simis'].apply(lambda x: x[2])
    tmp['simi_scores_std'] = tmp['simis'].apply(lambda x: x[3])
    tmp['simi_scores_mean'] = tmp['simis'].apply(lambda x: x[4])
    tmp['simi_scores_median'] = tmp['simis'].apply(lambda x: x[5])
    tmp['simi_scores_percentile_25'] = tmp['simis'].apply(lambda x: x[6])
    tmp['simi_scores_percentile_75'] = tmp['simis'].apply(lambda x: x[7])
    tmp['simi_scores_percentile_5'] = tmp['simis'].apply(lambda x: x[8])
    tmp['simi_scores_percentile_95'] = tmp['simis'].apply(lambda x: x[9])
    df_raw_tx = df_raw_tx.merge(tmp, on=['userId', 'itemId'], how='left')

    # step 5, add word emb
    df_raw_tx['w2v'] = df_raw_tx.apply(
        lambda x: get_wv(model_01.wv, key=x['itemId'], emb_dim=EMBDIM), axis=1)
    df_raw_tx = explode_emb(df_raw_tx, 'w2v', emb_dim=EMBDIM)
    del df_raw_tx['itemIds'], df_raw_tx['simis'], df_raw_tx['w2v']
    gc.collect()
    df_raw_tx = reduce_mem_usage(df_raw_tx)
    return df_raw_tx


def train_lgb(train_x_raw,
              train_y_raw,
              test_x_raw,
              params,
              categorical_feature=[],
              nfolds=10,
              num_boost_round=10000):
    print(params)

    train_x = train_x_raw.copy()
    train_y = train_y_raw.copy()
    test_x = test_x_raw.copy()

    train_x[categorical_feature] = train_x[categorical_feature].astype(
        'category')
    test_x[categorical_feature] = test_x[categorical_feature].astype(
        'category')

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=2022)
    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f">>> fold {i+1}")
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[
            train_index], train_x.iloc[valid_index], train_y[valid_index]
        dtrain = lgb.Dataset(trn_x, label=trn_y)
        dvalid = lgb.Dataset(val_x, label=val_y)

        model = lgb.train(params,
                          train_set=dtrain,
                          num_boost_round=num_boost_round,
                          valid_sets=[dtrain, dvalid],
                          categorical_feature=categorical_feature,
                          verbose_eval=100,
                          early_stopping_rounds=200)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        train[valid_index] = val_pred
        test += test_pred / nfolds
        cv_scores.append(roc_auc_score(val_y, val_pred))
        print(cv_scores)
    print("cv_scores:", cv_scores)
    print("mean+-std:", np.mean(cv_scores), "+-", np.std(cv_scores))
    return train, test, model


def get_submit_file(test_file, pred, default_val=0.0):
    dict_df = dict(zip(pred['userId'] + '_' + pred['itemId'], pred['score']))
    submit_file = test_file.copy()
    submit_file['itemIds'] = submit_file['itemIds'].apply(
        lambda x: x.split(','))
    submit_file = submit_file.explode('itemIds')
    submit_file['key'] = submit_file['userId'] + '_' + submit_file['itemIds']
    submit_file['score'] = submit_file.apply(
        lambda x: dict_df.get(x['key'], default_val), axis=1)
    submit_file = submit_file[['userId', 'itemIds', 'score']]
    submit_file.columns = ['userId', 'itemId', 'score']
    submit_file = submit_file.groupby('userId', group_keys=False).apply(
        lambda x: x.sort_values('score', ascending=False)).reset_index(
            drop=True)
    return submit_file


def make_pred(mkt, test_run_path_dict, fea_sub_dir):
    pkl_dir = os.path.join(fea_sub_dir, mkt, 'test_pred_all.pkl')
    final_dir = os.path.join(fea_sub_dir, mkt, 'test_pred_new.tsv')
    default_val = 0.0
    if ('i2i_bfs_novalid_maxpooling' in fea_sub_dir) or ('i2ibfssubmit'
                                                         in fea_sub_dir):
        default_val = -10.0
    print(f">>> Predict for {mkt}, test_run_path: {test_run_path_dict[mkt]}")
    print(f">>> pkl_dir: {pkl_dir}")
    print(f">>> final_dir: {final_dir}")
    new_test_run = pd.read_csv(test_run_path_dict[mkt],
                               sep='\t',
                               header=None,
                               names=['userId', 'itemIds'])
    test_pred_all = Cache.load_pkl(pkl_dir)
    submit_test_new = get_submit_file(test_file=new_test_run,
                                      pred=test_pred_all,
                                      default_val=default_val)
    submit_test_new.to_csv(final_dir, sep="\t", index=False)


def get_all_sub_dirs(feature_file_dir):
    if not feature_file_dir.endswith("/"):
        raise
    all_sub_dirs = glob.glob(feature_file_dir + "bsl_v2") + glob.glob(
        feature_file_dir + "gcntune1ksubmit_withoutqrel") + glob.glob(
            feature_file_dir +
            "ggcn*") + glob.glob(feature_file_dir + "i2i*") + glob.glob(
                feature_file_dir + "icf*") + glob.glob(feature_file_dir +
                                                       "swing*")
    print("len(all_sub_dirs): ", len(all_sub_dirs))
    print("all_sub_dirs: \n", all_sub_dirs)
    all_sub_dirs = [
        col for col in all_sub_dirs if 'ggcn_t_s0__main_64x' not in col
    ]
    return all_sub_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_pretrain', type=int, default=1)
    parser.add_argument('--retrain_all_retrieval', type=int, default=0)
    parser.add_argument('--t1_test_run_path', type=str, default='NONE_NONE')
    parser.add_argument('--t2_test_run_path', type=str, default='NONE_NONE')

    ARGS = parser.parse_args()
    print("=====" * 20)
    for k, v in ARGS.__dict__.items():
        print(f">>> {k}: {v}")
    print("=====" * 20)

    use_pretrain = ARGS.use_pretrain
    retrain_all_retrieval = ARGS.retrain_all_retrieval
    assert use_pretrain in [0, 1]
    assert retrain_all_retrieval in [0, 1]
    t1_test_run_path = ARGS.t1_test_run_path
    t2_test_run_path = ARGS.t2_test_run_path
    test_run_path_dict = {"t1": t1_test_run_path, "t2": t2_test_run_path}

    if retrain_all_retrieval:
        feature_file_dir = "/home/workspace/OUTPUT/00_NEW/"
    else:
        feature_file_dir = "/home/workspace/OUTPUT/pretrain_features/"

    if (t1_test_run_path == 'NONE_NONE') or (t2_test_run_path == 'NONE_NONE'):
        print("Make prediction for old test_run.tsv!!!")
        new_test_run_pred_flag = False
        test_pred_filenm = "test_pred.tsv"
    else:
        print("Make prediction for new test_run.tsv!!!")
        all_sub_dirs = get_all_sub_dirs(feature_file_dir)
        for fea_sub_dir in all_sub_dirs:
            make_pred(mkt='t1',
                      test_run_path_dict=test_run_path_dict,
                      fea_sub_dir=fea_sub_dir)
            make_pred(mkt='t2',
                      test_run_path_dict=test_run_path_dict,
                      fea_sub_dir=fea_sub_dir)
        print("Done! Finish making retrieval predictions for new test_run!")
        new_test_run_pred_flag = True
        test_pred_filenm = "test_pred_new.tsv"

    if use_pretrain and (
            not new_test_run_pred_flag):  # old test + use_pretrain
        # cached data v2
        t1_df_all = Cache.load_pkl(
            "/home/workspace/OUTPUT/cached_data/CACHE_data_t1_df__final.pkl")
        t2_df_all = Cache.load_pkl(
            "/home/workspace/OUTPUT/cached_data/CACHE_data_t2_df__final.pkl")
    else:
        # gen feas all
        t1_df_all = gen_feas_all(target_market='t1',
                                 new_test_run_path=t1_test_run_path)
        t2_df_all = gen_feas_all(target_market='t2',
                                 new_test_run_path=t2_test_run_path)

    id_map_all = {'t1': {}, 't2': {}}
    for col in ['userId', 'itemId']:
        id_unique_t1 = list(t1_df_all[col].unique())
        id_map_unique_t1 = dict(zip(id_unique_t1, range(len(id_unique_t1))))
        t1_df_all[f'{col}_feature'] = t1_df_all[col].apply(
            lambda x: id_map_unique_t1[x])
        id_map_all['t1'][col] = id_map_unique_t1

        id_unique_t2 = list(t2_df_all[col].unique())
        id_map_unique_t2 = dict(zip(id_unique_t2, range(len(id_unique_t2))))
        t2_df_all[f'{col}_feature'] = t2_df_all[col].apply(
            lambda x: id_map_unique_t2[x])
        id_map_all['t2'][col] = id_map_unique_t2

    valid_t1_df = t1_df_all[~t1_df_all.label.isnull()].reset_index(drop=True)
    test_t1_df = t1_df_all[t1_df_all.label.isnull()].reset_index(drop=True)
    valid_t2_df = t2_df_all[~t2_df_all.label.isnull()].reset_index(drop=True)
    test_t2_df = t2_df_all[t2_df_all.label.isnull()].reset_index(drop=True)

    T1_feature_file_list = [
        "gcntune1ksubmit_withoutqrel", "i2ibfssubmit", "bsl_v2",
        "ggcn_t_s0__main"
    ]  # ggcn_t_s0__main | ggcn_t_s0__main_64x

    T2_feature_file_list = [
        "swing_v2_train_only_no_t1t2_cross", "swing_v2_train_only_t1t2_cross",
        "i2i_bfs_novalid_maxpooling"
    ]

    target_market = 't1'
    for idx, colnm in enumerate(T1_feature_file_list):
        tmp_val = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                           target_market, 'valid_pred.tsv'),
                              sep='\t')
        tmp_test = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                            target_market, test_pred_filenm),
                               sep='\t')
        tmp_val = tmp_val.rename(columns={'score': colnm})
        tmp_test = tmp_test.rename(columns={'score': colnm})

        if colnm == 'ggcn_t_s0__main_64x':
            tmp_val = tmp_val.rename(
                columns={'ggcn_t_s0__main_64x': 'ggcn_t_s0__main'})
            tmp_test = tmp_test.rename(
                columns={'ggcn_t_s0__main_64x': 'ggcn_t_s0__main'})

        valid_t1_df = valid_t1_df.merge(tmp_val,
                                        on=['userId', 'itemId'],
                                        how='left')
        test_t1_df = test_t1_df.merge(tmp_test,
                                      on=['userId', 'itemId'],
                                      how='left')

    target_market = 't2'
    for idx, colnm in enumerate(T2_feature_file_list):
        tmp_val = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                           target_market, 'valid_pred.tsv'),
                              sep='\t')
        tmp_test = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                            target_market, test_pred_filenm),
                               sep='\t')
        tmp_val = tmp_val.rename(columns={'score': colnm})
        tmp_test = tmp_test.rename(columns={'score': colnm})
        valid_t2_df = valid_t2_df.merge(tmp_val,
                                        on=['userId', 'itemId'],
                                        how='left')
        test_t2_df = test_t2_df.merge(tmp_test,
                                      on=['userId', 'itemId'],
                                      how='left')

    # itemcf
    for idx, (i, j) in enumerate(zip(range(1, 21, 2), range(2, 21, 2))):
        val_feas = [f'icf_v2_{str(i).zfill(3)}', f'icf_v3_{str(i).zfill(3)}']
        test_feas = [f'icf_v2_{str(j).zfill(3)}', f'icf_v3_{str(j).zfill(3)}']
        print(val_feas)
        print(test_feas)

        for colnm_val in val_feas:
            if 'icf_v2_' in colnm_val:
                colnm = f'icf_v2__{idx}'
            elif 'icf_v3_' in colnm_val:
                colnm = f'icf_v3__{idx}'
            else:
                raise

            print(f"colnm:{colnm}")

            # t1
            target_market = 't1'
            tmp_val = pd.read_csv(os.path.join(feature_file_dir, colnm_val,
                                               target_market,
                                               'valid_pred.tsv'),
                                  sep='\t')

            tmp_val = tmp_val.rename(columns={'score': colnm})
            #         tmp_test = tmp_test.rename(columns={'score':colnm})

            valid_t1_df = valid_t1_df.merge(tmp_val,
                                            on=['userId', 'itemId'],
                                            how='left')
            # t2
            target_market = 't2'

            tmp_val = pd.read_csv(os.path.join(feature_file_dir, colnm_val,
                                               target_market,
                                               'valid_pred.tsv'),
                                  sep='\t')

            tmp_val = tmp_val.rename(columns={'score': colnm})
            #         tmp_test = tmp_test.rename(columns={'score':colnm})

            valid_t2_df = valid_t2_df.merge(tmp_val,
                                            on=['userId', 'itemId'],
                                            how='left')

        for colnm_test in test_feas:
            if 'icf_v2_' in colnm_test:
                colnm = f'icf_v2__{idx}'
            elif 'icf_v3_' in colnm_val:
                colnm = f'icf_v3__{idx}'
            else:
                raise

            print(f"colnm:{colnm}")
            # t1
            target_market = 't1'
            tmp_test = pd.read_csv(os.path.join(feature_file_dir, colnm_test,
                                                target_market,
                                                test_pred_filenm),
                                   sep='\t')
            tmp_test = tmp_test.rename(columns={'score': colnm})
            test_t1_df = test_t1_df.merge(tmp_test,
                                          on=['userId', 'itemId'],
                                          how='left')
            # t2
            target_market = 't2'
            tmp_test = pd.read_csv(os.path.join(feature_file_dir, colnm_test,
                                                target_market,
                                                test_pred_filenm),
                                   sep='\t')
            tmp_test = tmp_test.rename(columns={'score': colnm})
            test_t2_df = test_t2_df.merge(tmp_test,
                                          on=['userId', 'itemId'],
                                          how='left')

    # lgcn
    feature_file_list = [
        'ggcn_t_s0',
        'ggcn_t_s0_use_valid',
        'ggcn_t_s1',
        'ggcn_t_s1_use_valid',
        'ggcn_t_s1s2',
        'ggcn_t_s1s2_use_valid',
        'ggcn_t_s1s2s3',
        'ggcn_t_s1s2s3_use_valid',
        'ggcn_t_s1s3',
        'ggcn_t_s1s3_use_valid',
        'ggcn_t_s2',
        'ggcn_t_s2_use_valid',
        'ggcn_t_s2s3',
        'ggcn_t_s2s3_use_valid',
        'ggcn_t_s3',
        'ggcn_t_s3_use_valid',
    ]

    target_market = 't1'
    for idx, colnm in enumerate(feature_file_list):
        if 'valid' in colnm:

            tmp_test = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                                target_market,
                                                test_pred_filenm),
                                   sep='\t')
            _colnm_name = colnm.replace('_use_valid', '')
            tmp_test = tmp_test.rename(columns={'score': _colnm_name})
            test_t1_df = test_t1_df.merge(tmp_test,
                                          on=['userId', 'itemId'],
                                          how='left')

        else:
            tmp_val = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                               target_market,
                                               'valid_pred.tsv'),
                                  sep='\t')
            tmp_val = tmp_val.rename(columns={'score': colnm})
            valid_t1_df = valid_t1_df.merge(tmp_val,
                                            on=['userId', 'itemId'],
                                            how='left')

    target_market = 't2'
    for idx, colnm in enumerate(feature_file_list):
        if 'valid' in colnm:
            tmp_test = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                                target_market,
                                                test_pred_filenm),
                                   sep='\t')
            _colnm_name = colnm.replace('_use_valid', '')
            tmp_test = tmp_test.rename(columns={'score': _colnm_name})
            test_t2_df = test_t2_df.merge(tmp_test,
                                          on=['userId', 'itemId'],
                                          how='left')

        else:
            tmp_val = pd.read_csv(os.path.join(feature_file_dir, colnm,
                                               target_market,
                                               'valid_pred.tsv'),
                                  sep='\t')
            tmp_val = tmp_val.rename(columns={'score': colnm})
            valid_t2_df = valid_t2_df.merge(tmp_val,
                                            on=['userId', 'itemId'],
                                            how='left')

    params = conf_all.TREE_CONF
    print(params)

    t1_val_pred, t1_test_pred, model_t1 = train_lgb(
        valid_t1_df[T1_FEATURES_NEW_ALL],
        valid_t1_df['label'],
        test_t1_df[T1_FEATURES_NEW_ALL],
        params=params,
        categorical_feature=[])

    t2_val_pred, t2_test_pred, model_t2 = train_lgb(
        valid_t2_df[T2_FEATURES_NEW_ALL],
        valid_t2_df['label'],
        test_t2_df[T2_FEATURES_NEW_ALL],
        params=params,
        categorical_feature=[])

    valid_t1_df['score'] = t1_val_pred
    valid_t1_df = valid_t1_df.sort_values(['userId', 'score'], ascending=False)
    recom_t1_df = valid_t1_df.groupby(['userId'
                                       ])['itemId'].agg(list).reset_index()
    recom_t1_df.columns = ['userId', 'pred_itemIds']

    valid_t2_df['score'] = t2_val_pred
    valid_t2_df = valid_t2_df.sort_values(['userId', 'score'], ascending=False)
    recom_t2_df = valid_t2_df.groupby(['userId'
                                       ])['itemId'].agg(list).reset_index()
    recom_t2_df.columns = ['userId', 'pred_itemIds']

    valid_qrel_t1 = get_data_single('t1')['valid_qrel']
    valid_qrel_t2 = get_data_single('t2')['valid_qrel']

    recom_t1_df = recom_t1_df.merge(valid_qrel_t1, on='userId', how='left')
    NDCG = 0
    for items in recom_t1_df[['pred_itemIds', 'itemId']].values:
        l1 = items[0][:10]
        l2 = [items[1]]
        NDCG += getNDCG(l1, l2)
    NDCG = NDCG / len(valid_qrel_t1)
    print('T1 NDCG : ', NDCG)
    recom_t2_df = recom_t2_df.merge(valid_qrel_t2, on='userId', how='left')
    NDCG = 0
    for items in recom_t2_df[['pred_itemIds', 'itemId']].values:
        l1 = items[0][:10]
        l2 = [items[1]]
        NDCG += getNDCG(l1, l2)
    NDCG = NDCG / len(valid_qrel_t2)
    print('T2 NDCG : ', NDCG)

    OUTPUT_DIRNM = "FINAL"
    OUTPUT_TMP_DIR = "SUB"
    OUTPUT_PATH = os.path.join(OUTPUT_BASEDIR, OUTPUT_TMP_DIR, OUTPUT_DIRNM)
    OUTPUT_PATH_t1 = os.path.join(OUTPUT_PATH, 't1')
    OUTPUT_PATH_t2 = os.path.join(OUTPUT_PATH, 't2')
    mkdirs(OUTPUT_PATH)
    mkdirs(OUTPUT_PATH_t1)
    mkdirs(OUTPUT_PATH_t2)
    OUTPUT_PATH_dict = {'t1': OUTPUT_PATH_t1, 't2': OUTPUT_PATH_t2}

    # t1
    test_t1_df['score'] = t1_test_pred
    test_t1_df = test_t1_df.sort_values(['userId', 'score'], ascending=False)
    test_t1_df[['userId', 'itemId',
                'score']].to_csv(os.path.join(OUTPUT_PATH_t1,
                                              test_pred_filenm),
                                 sep='\t',
                                 index=False)
    valid_t1_df[['userId', 'itemId',
                 'score']].to_csv(f'{OUTPUT_PATH_t1}/valid_pred.tsv',
                                  sep='\t',
                                  index=False)

    # t2
    test_t2_df['score'] = t2_test_pred
    test_t2_df = test_t2_df.sort_values(['userId', 'score'], ascending=False)
    test_t2_df[['userId', 'itemId',
                'score']].to_csv(os.path.join(OUTPUT_PATH_t2,
                                              test_pred_filenm),
                                 sep='\t',
                                 index=False)
    valid_t2_df[['userId', 'itemId',
                 'score']].to_csv(f'{OUTPUT_PATH_t2}/valid_pred.tsv',
                                  sep='\t',
                                  index=False)

    print(bash2py(f"ls {OUTPUT_PATH}"))
    cmd = f"cd {OUTPUT_PATH} && zip -r submission.zip *"
    bash2py(cmd)
    print(bash2py(f"ls {OUTPUT_PATH}"))