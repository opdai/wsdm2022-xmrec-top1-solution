walk_length = 100  # 句子长度
num_walks = 64  # 图里每个节点游走生成句子数
workers = 64  # 进程数
_p = 20  # 值越大，越不容易回到前一个节点
_q = 13  # 值越大，越容易留在邻居，是BFS（结构性）；否则是DFS（同质性）

vector_size = 32
window = 50
sg = 1
hs = 0
negative = 10
