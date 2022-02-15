# coding:utf-8
import config
import numpy as np
import pandas as pd
import torch
import utils
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score

from pytz import timezone
from datetime import datetime

UTC8 = timezone('Asia/Shanghai')


def time_now():
    return datetime.now(UTC8).strftime('%Y%m%d%H%M%S')


CORES = 16


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(config.device)
    posItems = posItems.to(config.device)
    negItems = negItems.to(config.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if config.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / config.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in config.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def predict(dataset, Recmodel, tmarket='t1t2', pattern=''):
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    testDict = dataset.testDict

    users_to_test = []
    if tmarket == 't1t2':
        # fix test user to t1t2 valid&run
        _test_set_path = '/home/workspace/DATA/t1t2/valid_qrel.tsv'
        users_to_test = pd.read_csv(_test_set_path, sep='\t', header=0)
        users_to_test = users_to_test['userId'].unique().tolist()
        print(f'->[testUser]picking t1t2 user to predict')
        print(f'->[testUser]num:{len(users_to_test)} path:{_test_set_path}')
    elif tmarket == 't1':
        users_to_test = list(testDict.keys())
        print(f'->[testUser]picking t1 user to predict')
        print(f'->[testUser]num:{len(users_to_test)}')
    elif tmarket == 't2':
        users_to_test = list(testDict.keys())
        print(f'->[testUser]picking t2 user to predict')
        print(f'->[testUser]num:{len(users_to_test)}')

    u_batch_size = len(users_to_test)
    print(f'->[testUser]set u_batch_size = {u_batch_size}')

    ratings = []

    with torch.no_grad():
        users = users_to_test
        ratinglist = []

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            rating = rating.cpu().numpy()
            ratinglist.append(rating)

    ratings = np.concatenate(ratinglist, axis=0)
    print(f'->[predict]result shape:{ratings.shape}. saving result..')

    # saving result
    output_dir = './test_result'
    output_fname = f'{output_dir}/test_result_{pattern}.txt'
    assert len(users_to_test) == len(ratings), f'users_to_test={len(users_to_test)}, ratings={len(ratings)}'
    with open(output_fname, 'w') as wf:
        for uid_idx, pred_res in zip(users_to_test, ratings):
            wf.write('{}\t{}\n'.format(uid_idx, list(pred_res)))
    print(f'->[Done]saving as {output_fname}')


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = config.config['test_u_batch_size']
    testDict = dataset.testDict
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(config.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(config.topks)),
               'recall': np.zeros(len(config.topks)),
               'ndcg': np.zeros(len(config.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if config.tensorboard:
            w.add_scalars(f'Test/Recall@{config.topks}',
                          {str(config.topks[i]): results['recall'][i] for i in range(len(config.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{config.topks}',
                          {str(config.topks[i]): results['precision'][i] for i in range(len(config.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{config.topks}',
                          {str(config.topks[i]): results['ndcg'][i] for i in range(len(config.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
