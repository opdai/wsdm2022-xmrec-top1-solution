import subprocess
from . import logger
import os
import psutil


def mkdirs(dir2make):
    if isinstance(dir2make, list):
        for i_dir in dir2make:
            if not os.path.exists(i_dir):
                os.makedirs(i_dir)
    elif isinstance(dir2make, str):
        if not os.path.exists(dir2make):
            os.makedirs(dir2make)
    else:
        raise ValueError("dir2make should be string or list type.")



def bash2py(shell_command, split=True):
    """
    Args:
        shell_command: str, shell_command
        No capture_output arg for < py3.7 !
    example:
        bash2py('du -sh')    
    """
    res = subprocess.run(shell_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if res.returncode == 0:
        logger.info(f"Execute <{shell_command}> successfully!")
    else:
        raise Exception(f"ERROR: {res.stderr.decode('utf-8')}")
    res = res.stdout.decode('utf-8').strip()  #.split('\n')
    if split:
        return res.split('\n')
    else:
        return res



def check_deepmatch_train_process(cmd):
    # cmd = "ps aux | grep python | grep deepmatch.py | grep -v grep | grep -v daily_train_deepmatch.py | awk '{print $2}'"
    ps_lst = bash2py(cmd)
    if len(ps_lst)>1:
        logger.info("More than one related Process is still running: ")
        logger.info("==="*10)
        for ipid in ps_lst:
            if not ipid:
                continue
            p = psutil.Process(int(ipid))
            logger.info(f"ipid:{ipid} || name:{p.name()} || cmdline:{p.cmdline()}")
        logger.info("==="*10)
        return False
    else:
        if not ps_lst[0]:
            logger.info("No such process!")
            return False
        else:
            logger.info("Process is still running!")
            p = psutil.Process(int(ps_lst[0]))
            logger.info(f"ipid:{ps_lst[0]} || name:{p.name()} || cmdline:{p.cmdline()}")
            return True
        