import logging

logging.basicConfig(
    format=
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_log(file_name, log_name, log_level=logging.INFO):
    logger = logging.getLogger(name=log_name)
    logger.setLevel(log_level)
    if len(logger.handlers): 
        return logger
    handler = logging.FileHandler(file_name)
    handler.setLevel(log_level)
    FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    DATEFMT = "[%Y-%m-%d %H:%M:%S]"
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger