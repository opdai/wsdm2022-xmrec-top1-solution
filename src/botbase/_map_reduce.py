import random
from . import fix_random


def get_n_parts_idx_lst(idx_lst,n_parts,do_shuffle=False):
    """split a list into n parts.
    idx_lst = list(range(20))
    n_parts = 5
    get_n_parts_idx_lst(idx_lst,n_parts)
    """
    idx_lst_lst = []
    if do_shuffle:
        fix_random(666)
        random.shuffle(idx_lst)
    num_in_one_part = len(idx_lst) // n_parts
    length_all = []
    for i in range(n_parts):
        if i == n_parts-1:
            idx_lst_tmp = idx_lst[i*num_in_one_part:]
        else:
            idx_lst_tmp = idx_lst[i*num_in_one_part:(i+1)*num_in_one_part]
        idx_lst_lst.append(idx_lst_tmp)
        length_all.append(len(idx_lst_tmp))
    print(f"length_all: {length_all}")
    return idx_lst_lst