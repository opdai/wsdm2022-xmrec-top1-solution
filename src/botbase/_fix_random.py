import os
import numpy as np
import random


def fix_random(seed=666, seed_torch=False, seed_tf=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if seed_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    if seed_tf:
        import tensorflow as tf
        try:
            tf.set_random_seed(seed)
        except:
            tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
