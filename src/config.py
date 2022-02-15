TREE_CONF = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'min_child_weight': 5,
    'num_leaves': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 4,
    'learning_rate': 0.05,
    'seed': 666,
    'n_jobs': 32,
    'verbose': -1 }

EMB_SIZE=32
EPOCHS=5
WINDOW=100
MIN_COUNT=1
SG=1
HS=1
NEGATIVE=10
N_JOBS=32
SEED=666