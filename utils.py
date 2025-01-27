import sys
import logging


def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Sets up a basic logger configuration.

    Args:
        name: The name of the logger.
        log_level: The logging level (e.g., logging.DEBUG, logging.INFO, etc.). Defaults to INFO.
        log_file: The path to a file for logging messages. Defaults to None (logging to console).
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
