# coding:utf-8
import os
import sys

sys.path.insert(0, '/home/workspace/')
if not os.path.exists('/home/workspace/src/retrieval/lgcn_v2/data'):
    os.mkdir('/home/workspace/src/retrieval/lgcn_v2/data')
if not os.path.exists('/home/workspace/src/retrieval/lgcn_v2/test_result'):
    os.mkdir('/home/workspace/src/retrieval/lgcn_v2/test_result')

import config
import utils
import dataloader
import model
import torch
import time
import Procedure
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from config import cprint
from pprint import pprint
from src.botbase import Cache
from lgcnv2Params import result_pair, data_combination

# ==============================
utils.set_seed(config.seed)
print(">>SEED:", config.seed)
# ==============================

print('===========config================')
pprint(config.config)
print("cores for test:", config.CORES)
print("comment:", config.comment)
print("tensorboard:", config.tensorboard)
print("LOAD:", config.LOAD)
print("Weight path:", config.PATH)
print("Test Topks:", config.topks)
print("using bpr loss")
print('===========end===================')


def load_target_valid_test(i=1):
    df_test_run = pd.read_csv(f'/home/workspace/DATA/t{i}/test_run.tsv', sep='\t', header=None,
                              names=['userId', 'itemId'])
    df_valid_run = pd.read_csv(f'/home/workspace/DATA/t{i}/valid_run.tsv', sep='\t', header=None,
                               names=['userId', 'itemId'])
    print(f'DATA/t{i}/test_run:{df_test_run.shape}, /DATA/t{i}/valid_run:{df_valid_run.shape}')

    # split candidate list into rows
    df_test_run['itemId'] = df_test_run['itemId'].apply(lambda t: t.split(','))
    df_valid_run['itemId'] = df_valid_run['itemId'].apply(lambda t: t.split(','))

    df_test_run = df_test_run.explode('itemId')
    df_valid_run = df_valid_run.explode('itemId')
    return df_valid_run, df_test_run


def _restore_score(t):
    try:
        score = t['scoredic'].get(t['itemId'], -10)
    except:
        score = -10
        print(f"err user:{t['userId']} item:{t['itemId']} scoredic:{t['scoredic']} "
              f"user isin vocab:{t['userId'] in pred_user_list} item isin vocab:{t['itemId'] in itemlist}")
    return score


def formup_submittion(src_df, global_pred_res_df, savename='default'):
    # join scoredic by userId
    _src_df = src_df.merge(global_pred_res_df, on='userId', how='left', validate='m:1')
    # pickup itemscore in scoredic by itemId
    _src_df['score'] = _src_df.apply(_restore_score, axis=1)
    # formatting
    _submit = _src_df.loc[:, ['userId', 'itemId', 'score']].sort_values(['userId', 'score'], ascending=False)

    _submit.to_csv(savename, sep='\t', index=False)
    print(f'->saving as {savename}:{_submit.shape}')


if __name__ == '__main__':

    """ --------------- train & inference --------------- """

    for data_path, train_setting in data_combination.items():
        print(f'->training with combination: {data_path}')
        de_train_epoch = train_setting['train_epoch']
        de_train_embsize = train_setting['latent_dim_rec']
        de_train_batchsize = train_setting['train_batch_size']

        if '__main' in data_path:
            config.config['keep_prob'] = 0.6
            utils.set_seed(666)
        else:
            config.config['keep_prob'] = 0.9
            utils.set_seed(782)

        config.config['latent_dim_rec'] = de_train_embsize
        config.config['bpr_batch_size'] = de_train_batchsize

        dataset = dataloader.Loader(path="./data/" + data_path, modeltag=data_path)
        Recmodel = model.LightGCN(config.config, dataset)
        Recmodel = Recmodel.to(config.device)
        bpr = utils.BPRLoss(Recmodel, config.config)

        # weight_file = utils.getFileName(pattern=data_path)
        # print(f"load and save to {weight_file}")
        # if config.LOAD:
        #     try:
        #         Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        #         config.cprint(f"loaded model weights from {weight_file}")
        #     except FileNotFoundError:
        #         print(f"{weight_file} not exists, start from beginning")
        Neg_k = 1

        # init tensorboard
        # if config.tensorboard:
        #     tfb_path = f'{config.BOARD_PATH}/{time.strftime("%m-%d-%Hh%Mm%Ss-")}{data_path}'
        #     w: SummaryWriter = SummaryWriter(tfb_path)
        #     config.cprint(f"[tfboard]{tfb_path}")
        # else:
        w = None
        config.cprint("[tfboard]not enable tensorflowboard")

        if config.INFERENCE:
            Procedure.predict(dataset, Recmodel, tmarket=data_path.split('_')[0], pattern=data_path)
            sys.exit()

        for epoch in range(de_train_epoch):
            start = time.time()
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            epend = time.time()

            if epoch % 20 == 0:
                cprint("[TEST]")
                ts = time.time()
                Procedure.Test(dataset, Recmodel, epoch, w, config.config['multicore'])

                print(
                    f'{data_path} EPOCH[{epoch + 1}/{de_train_epoch}] {output_information} '
                    f'traintime:{(epend - start) / 60.:.1f}min testtime:{time.time() - ts:.1f}s')
                # _weight_file = weight_file.replace('.pth.tar', f'_ep{epoch}.pth.tar')
                # torch.save(Recmodel.state_dict(), _weight_file)

        # predict submittion
        Procedure.predict(dataset, Recmodel, tmarket=data_path.split('_')[0], pattern=data_path)

    """ --------------- restore u-i ranking score --------------- """
    
    df_t1_valid, df_t1_test = load_target_valid_test(1)
    df_t2_valid, df_t2_test = load_target_valid_test(2)

    for t1_data_path, t2_data_path in result_pair:

        for use_valid_flag in [False, True]:
            if use_valid_flag:
                t1_data_path = f'{t1_data_path}_use_valid'
                t2_data_path = f'{t2_data_path}_use_valid'

            t1_res_path = f'./test_result/test_result_{t1_data_path}.txt'
            t2_res_path = f'./test_result/test_result_{t2_data_path}.txt'

            # Market T1
            _item_vocab = pd.read_csv(f'./data/{t1_data_path}/item_list.txt', sep=' ', header=0)
            _item_vocab = dict(zip(_item_vocab['org_id'], _item_vocab['remap_id']))
            _user_vocab = pd.read_csv(f'./data/{t1_data_path}/user_list.txt', sep=' ', header=0)
            _user_vocab = dict(zip(_user_vocab['org_id'], _user_vocab['remap_id']))
            itemlist = list(_item_vocab.keys())
            userlist = list(_user_vocab.keys())
            print(f'itemlist:{len(itemlist)} userlist:{len(userlist)}')

            pred_res = pd.read_csv(t1_res_path, sep='\t', header=None, names=['userId', 'itemScoreList'])
            pred_res['itemScoreList'] = pred_res['itemScoreList'].apply(eval)
            pred_res['scoredic'] = pred_res['itemScoreList'].apply(lambda t: dict(zip(itemlist, t)))
            pred_res['userId'] = pred_res['userId'].apply(lambda t: userlist[int(t)])
            t1_pred_res_dic = pred_res.loc[:, ['userId', 'scoredic']]

            ttmp_1 = pred_res.explode('itemScoreList')
            ttmp_2 = pd.DataFrame(pred_res['userId'])
            ttmp_2.loc[:, 'itemId'] = str(itemlist)
            ttmp_2['itemId'] = ttmp_2['itemId'].apply(eval)
            ttmp_2 = ttmp_2.explode('itemId')
            ttmp_2 = ttmp_2.rename(columns={'userId': 'userId_'})

            assert ttmp_1.shape[0] == ttmp_2.shape[0]
            t1_global_score = pd.concat([ttmp_1, ttmp_2], axis=1)
            t1_global_score = t1_global_score.loc[:, ['userId', 'itemId', 'itemScoreList']].rename(
                columns={'itemScoreList': 'score'})
            print(t1_global_score.shape)

            # Market T2
            _item_vocab = pd.read_csv(f'./data/{t2_data_path}/item_list.txt', sep=' ', header=0)
            _item_vocab = dict(zip(_item_vocab['org_id'], _item_vocab['remap_id']))
            _user_vocab = pd.read_csv(f'./data/{t2_data_path}/user_list.txt', sep=' ', header=0)
            _user_vocab = dict(zip(_user_vocab['org_id'], _user_vocab['remap_id']))
            itemlist = list(_item_vocab.keys())
            userlist = list(_user_vocab.keys())
            print(f'itemlist:{len(itemlist)} userlist:{len(userlist)}')

            pred_res = pd.read_csv(t2_res_path, sep='\t', header=None, names=['userId', 'itemScoreList'])
            pred_res['itemScoreList'] = pred_res['itemScoreList'].apply(eval)
            pred_res['scoredic'] = pred_res['itemScoreList'].apply(lambda t: dict(zip(itemlist, t)))
            pred_res['userId'] = pred_res['userId'].apply(lambda t: userlist[int(t)])
            t2_pred_res_dic = pred_res.loc[:, ['userId', 'scoredic']]

            ttmp_1 = pred_res.explode('itemScoreList')
            ttmp_2 = pd.DataFrame(pred_res['userId'])
            ttmp_2.loc[:, 'itemId'] = str(itemlist)
            ttmp_2['itemId'] = ttmp_2['itemId'].apply(eval)
            ttmp_2 = ttmp_2.explode('itemId')
            ttmp_2 = ttmp_2.rename(columns={'userId': 'userId_'})

            assert ttmp_1.shape[0] == ttmp_2.shape[0]
            t2_global_score = pd.concat([ttmp_1, ttmp_2], axis=1)
            t2_global_score = t2_global_score.loc[:, ['userId', 'itemId', 'itemScoreList']].rename(
                columns={'itemScoreList': 'score'})
            print(t2_global_score.shape)

            # concat T1T2
            global_score = pd.concat([t1_global_score, t2_global_score], axis=0)
            print(global_score.shape)

            # always check if uid notin predict userlist
            pred_res_dic = pd.concat([t1_pred_res_dic, t2_pred_res_dic], axis=0)
            pred_user_list = pred_res_dic['userId'].unique().tolist()

            # save as features
            dirpattern = t1_data_path.split('_')
            dirpattern = 'ggcn_t_' + '_'.join(dirpattern[1:])
            dirpattern = '/home/workspace/OUTPUT/00_NEW/' + dirpattern
            if not os.path.exists('/home/workspace/OUTPUT/00_NEW/'):
                print(f'->Create directory: /home/workspace/OUTPUT/00_NEW/')
                os.mkdir('/home/workspace/OUTPUT/00_NEW/')
            if not os.path.exists(dirpattern):
                print(f'->Create directory: {dirpattern}')
                os.mkdir(f'{dirpattern}')
                os.mkdir(f'{dirpattern}/t1')
                os.mkdir(f'{dirpattern}/t2')
            formup_submittion(df_t1_valid, pred_res_dic, f'{dirpattern}/t1/valid_pred.tsv')
            formup_submittion(df_t1_test, pred_res_dic, f'{dirpattern}/t1/test_pred.tsv')
            formup_submittion(df_t2_valid, pred_res_dic, f'{dirpattern}/t2/valid_pred.tsv')
            formup_submittion(df_t2_test, pred_res_dic, f'{dirpattern}/t2/test_pred.tsv')

            # save as features
            dirpattern = t1_data_path.split('_')
            dirpattern = 'ggcn_t_' + '_'.join(dirpattern[1:])
            dirpattern = '/home/workspace/OUTPUT/00_NEW/' + dirpattern
            if not os.path.exists('/home/workspace/OUTPUT/00_NEW/'):
                print(f'->Create directory: /home/workspace/OUTPUT/00_NEW/')
                os.mkdir('/home/workspace/OUTPUT/00_NEW/')
            if not os.path.exists(dirpattern):
                print(f'->Create directory: {dirpattern}')
                os.mkdir(f'{dirpattern}')
                os.mkdir(f'{dirpattern}/t1')
                os.mkdir(f'{dirpattern}/t2')
            # global_score.to_csv(_save_path, header=True, index=False, seq='\t')
            Cache.dump_pkl(t1_global_score, f'{dirpattern}/t1/test_pred_all.pkl')
            Cache.dump_pkl(t2_global_score, f'{dirpattern}/t2/test_pred_all.pkl')
