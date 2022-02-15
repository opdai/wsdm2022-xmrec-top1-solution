import math
import os
import zipfile
from tqdm import tqdm

# Bi-Graph
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
                sim_item[item][relate_item] += 1/ (math.log(len(users)+1) * math.log(tmp_len+1))
            
    return sim_item, user_item_dict

### S123的全部数据都加入训练集
def load_source_trainset(i=1):
    df_rating = pd.read_csv(f'./DATA/s{i}/train.tsv', sep='\t', header=0)
    df_rating5 = pd.read_csv(f'./DATA/s{i}/train_5core.tsv', sep='\t', header=0)
    df_qrel = pd.read_csv(f'./DATA/s{i}/valid_qrel.tsv', sep='\t', header=0)
    print(f'/DATA/s{i}:{df_rating.shape}, 5core:{df_rating5.shape}, qrel:{df_qrel.shape}')

    df = pd.concat([df_rating, df_rating5, df_qrel], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    print(f'\t[i=={i}]df.shape:{df.shape}')
    return df

### 对T1保留T1qrel作为训练收敛指标参考，T2qrel加入训练
def load_target_trainset(i=1, keep_qrel=1):
    append_list = []
    df_rating = pd.read_csv(f'./DATA/t{i}/train.tsv', sep='\t', header=0)
    df_rating5 = pd.read_csv(f'./DATA/t{i}/train_5core.tsv', sep='\t', header=0)
    if i == keep_qrel:
        print(f'/DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}')
        print(f'->keep {keep_qrel} qrel as validation')
        df = pd.concat([df_rating, df_rating5], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
    else:
        df_qrel = pd.read_csv(f'./DATA/t{i}/valid_qrel.tsv', sep='\t', header=0)
        print(f'/DATA/t{i}:{df_rating.shape}, 5core:{df_rating5.shape}, qrel:{df_qrel.shape}')
        df = pd.concat([df_rating, df_rating5, df_qrel], axis=0).drop_duplicates(['userId', 'itemId'], keep='first')
        
    print(f'\t[i=={i}]df.shape:{df.shape}')
    return df

def load_target_valid_test(i=1):
    df_test_run = pd.read_csv(f'./DATA/t{i}/test_run.tsv', sep='\t', header=None, names=['userId', 'itemId'])
    df_valid_run = pd.read_csv(f'./DATA/t{i}/valid_run.tsv', sep='\t', header=None, names=['userId', 'itemId'])
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

def zip_submittion(srcpath, outputpath):
    fzip = zipfile.ZipFile(outputpath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirs, fnames in os.walk(srcpath):
        fpath = path.replace(srcpath, '')
        for fname in fnames:
            fzip.write(os.path.join(path, fname), os.path.join(fpath, fname))
    fzip.close()

data_dic = {
    's1': load_source_trainset(1),
    's2': load_source_trainset(2),
    's3': load_source_trainset(3),
    
    't1': load_target_trainset(1, keep_qrel=1),
    't2': load_target_trainset(2, keep_qrel=2),
    
    't1_valid': pd.read_csv(f'./DATA/t1/valid_qrel.tsv', sep='\t', header=0),
    't2_valid': pd.read_csv(f'./DATA/t2/valid_qrel.tsv', sep='\t', header=0),
}

combination = [
    's0',
    's1',
    's1s2',
    's1s2s3',
    's1s3',
    's2',
    's2s3',
    's3',
]

df_t1_valid, df_t1_test = load_target_valid_test(1)
df_t2_valid, df_t2_test = load_target_valid_test(2)

# always check if uid notin predict userlist
pred_t1_user_list = pd.read_csv(f'./DATA/t1/valid_qrel.tsv', sep='\t', header=0)['userId'].unique().tolist()
pred_t2_user_list = pd.read_csv(f'./DATA/t2/valid_qrel.tsv', sep='\t', header=0)['userId'].unique().tolist()
print(f'->pred_user_list t1:{len(pred_t1_user_list)} t2:{len(pred_t2_user_list)}')

for use_valid_flag in [False, True]:
    for _market in combination:
        print('>' * 50)
        print(f'->combination:{_market}')
        
        # calc recall dic for every user in userlist
        uid2recalldic = {}
        
        print('---------------------------- market T1 ----------------------------')
        df_list = []
        df_list.append(data_dic['t1'])
        df_list.append(data_dic['t2_valid'])
        if 's1' in _market:
            df_list.append(data_dic['s1'])
            print('\t->append s1')
        if 's2' in _market:
            df_list.append(data_dic['s2'])
            print('\t->append s2')
        if 's3' in _market:
            df_list.append(data_dic['s3'])
            print('\t->append s3')
        if use_valid_flag:
            df_list.append(data_dic['t1_valid'])
            print('\t->append t1train&valid + t2valid')
        else:
            print('\t->append t1train + t2valid')
        
        # concat trainingdata
        total_qrel_df = pd.concat(df_list).drop_duplicates(['userId', 'itemId'], keep='last')
        print(f'->[T1]total_qrel_df:{total_qrel_df.shape}')
        
        # bi-Graph
        i2i_sim_dic, u2i_dict = get_sim_item(total_qrel_df, 'userId', 'itemId')
        
        for _uid in tqdm(pred_t1_user_list):
            _global_recall_dict = {}
            _user_sequence = u2i_dict.get(_uid, {})
            for _item in _user_sequence:
                _local_rec_dic = i2i_sim_dic.get(_item, {})
                for _k, _v in _local_rec_dic.items():
                    # add up all recallitem's score
                    if _k in _global_recall_dict:
                        _global_recall_dict[_k] += _v
                    else:
                        _global_recall_dict[_k] = _v
            uid2recalldic[_uid] = _global_recall_dict
        print(f'->[T1]done recall for {len(uid2recalldic)} user in testset')
        
        
        print('---------------------------- market T2 ----------------------------')
        df_list = []
        df_list.append(data_dic['t2'])
        df_list.append(data_dic['t1_valid'])
        if 's1' in _market:
            df_list.append(data_dic['s1'])
            print('\t->append s1')
        if 's2' in _market:
            df_list.append(data_dic['s2'])
            print('\t->append s2')
        if 's3' in _market:
            df_list.append(data_dic['s3'])
            print('\t->append s3')
        if use_valid_flag:
            df_list.append(data_dic['t2_valid'])
            print('\t->append t2train&valid + t1valid')
        else:
            print('\t->append t2train + t1valid')
        
        # concat trainingdata
        total_qrel_df = pd.concat(df_list).drop_duplicates(['userId', 'itemId'], keep='last')
        print(f'->[T2]total_qrel_df:{total_qrel_df.shape}')
        
        # bi-Graph
        i2i_sim_dic, u2i_dict = get_sim_item(total_qrel_df, 'userId', 'itemId')
        
        for _uid in tqdm(pred_t2_user_list):
            _global_recall_dict = {}
            _user_sequence = u2i_dict.get(_uid, {})
            for _item in _user_sequence:
                _local_rec_dic = i2i_sim_dic.get(_item, {})
                for _k, _v in _local_rec_dic.items():
                    # add up all recallitem's score
                    if _k in _global_recall_dict:
                        _global_recall_dict[_k] += _v
                    else:
                        _global_recall_dict[_k] = _v
            uid2recalldic[_uid] = _global_recall_dict
        print(f'->[T2]done recall for {len(uid2recalldic)} user in testset')
        
        # saving
        spath = '/home/workspace/wsdm22_cup_xmrec/gcn_torch_org/fused_submit'
        formup_submittion_bigraph(df_t1_valid, uid2recalldic, savename=f'{spath}/t1/valid_pred.tsv')
        formup_submittion_bigraph(df_t1_test, uid2recalldic, savename=f'{spath}/t1/test_pred.tsv')
        formup_submittion_bigraph(df_t2_valid, uid2recalldic, savename=f'{spath}/t2/valid_pred.tsv')
        formup_submittion_bigraph(df_t2_test, uid2recalldic, savename=f'{spath}/t2/test_pred.tsv')
        
        sfilename = f"/home/workspace/wsdm22_cup_xmrec/gcn_torch_org/bigraph_{_market}{'_use_valid' if use_valid_flag else ''}.zip"
        zip_submittion(spath, sfilename)
        print(f'->[Done]saved as {sfilename}')
