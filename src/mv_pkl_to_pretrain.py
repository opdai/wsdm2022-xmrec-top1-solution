import sys
sys.path.insert(0, '/home/workspace')
import os
import glob
from src.botbase import bash2py


all_pkl = glob.glob("/home/workspace/OUTPUT/00_NEW/*/*/*.pkl")
print(all_pkl)
print(len(all_pkl))

for f1 in all_pkl:
    f2 = f1.replace("00_NEW","pretrain_features")
    cmd = f"cp {f1} {f2}"
    bash2py(cmd)
