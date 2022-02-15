from ..botbase import mkdirs

DATA_BASEDIR = "/home/workspace/DATA/"
OUTPUT_BASEDIR = "/home/workspace/OUTPUT/"
OUTPUT_BASEDIR_00_NEW = "/home/workspace/OUTPUT/00_NEW"

# paths of new unseen test_run set, for example:
# t1_test_run_path = '/home/workspace/DATA/t1/test_run.tsv'
# t2_test_run_path = '/home/workspace/DATA/t2/test_run.tsv'
# 'NONE_NONE' is the default value for no more new test_run set testing
t1_test_run_path = 'NONE_NONE' 
t2_test_run_path = 'NONE_NONE'


dirs_all = [DATA_BASEDIR, OUTPUT_BASEDIR, OUTPUT_BASEDIR_00_NEW]
for col in dirs_all:
    mkdirs(col)