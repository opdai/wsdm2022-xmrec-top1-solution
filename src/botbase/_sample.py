import random

def gen_nunique_nums(range_min,range_max,n2return):
    """
    gen_nunique_nums(0,23,5)
    """
    if not n2return:
        return []
    lst = []
    if range_max-range_min+1 <= n2return:
        print("range is not enough to sample.")
        raise ValueError("range is not enough to sample.")
        # return list(range(range_min,range_max+1))
    while 1:
        idxint = random.randint(range_min,range_max)
        if idxint not in lst:
            lst.append(idxint)
        if len(lst) == n2return:
            break
    return lst