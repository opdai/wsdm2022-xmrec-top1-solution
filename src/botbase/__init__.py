__all__ = ['fix_random', 'logger', 'custom_log', 'mkdirs', 'bash2py', 'isnotebook','Cache', 'FilesysHelper','gen_nunique_nums','check_process', 'get_n_parts_idx_lst']

from ._fix_random import fix_random
from ._log import logger, custom_log
from ._shell_helper import mkdirs, bash2py
from .config import BASE_DIR_abs as BASE_DIR 
from ._cache import Cache
from ._filesys_utils import FilesysHelper
from ._sample import gen_nunique_nums
from ._sys_utils import check_process
from ._map_reduce import get_n_parts_idx_lst

def func_isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

isnotebook = func_isnotebook()