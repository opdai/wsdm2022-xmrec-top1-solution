from . import bash2py,logger
import psutil

def check_process(includes,excludes=[]):
    """
    cmd = "ps aux | grep python | grep deepmatch.py | grep -v grep | grep -v daily_train_deepmatch.py | awk '{print $2}'"    
    check_process(includes=['python','deepmatch.py'], excludes=['daily_train_deepmatch.py'])
    """
    cmd = "ps aux"
    for ii in includes:
        cmd+=f" | grep {ii}"
    for ii in excludes:
        cmd+=f" | grep -v {ii}"
    cmd+=" | grep -v grep | awk '{print $2}'"

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
        return True
    else:
        if not ps_lst[0]:
            logger.info("No such process!")
            return False
        else:
            logger.info("Process is still running!")
            p = psutil.Process(int(ps_lst[0]))
            logger.info(f"ipid:{ps_lst[0]} || name:{p.name()} || cmdline:{p.cmdline()}")
            return True