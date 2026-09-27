import logging
import sys

def setup_logger (name = "miniserving", level = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        datefmt = "%H:%M:%S"
    )    
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()