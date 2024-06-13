import sys
import random
import numpy as np

# Log levels
LOG_LEVELS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
}


DEFAULT_LOG_LEVEL = "WARNING"
LOG_LEVEL = LOG_LEVELS.get(DEFAULT_LOG_LEVEL)


def set_log_level(log_level):
    """Set the global log level.

    Args:
        log_level (str): The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        None: This function does not return a value.
    """
    global LOG_LEVEL
    LOG_LEVEL = LOG_LEVELS.get(log_level.upper(), LOG_LEVELS.get(DEFAULT_LOG_LEVEL))


def set_seed(seed):
    """
    Set random seed for Blender, Python's random module, and numpy.
    """
    # Set random seed for Python's random module
    random.seed(seed)
    # Set random seed for numpy
    np.random.seed(seed)


def log(log_level, *args):
    """Wrapper for "print" that writes on stderr, without newline.

    Args:
        log_level (str): The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        args: list of arguments

    Returns:
        None: This function does not return a value.
    """
    if LOG_LEVELS.get(log_level.upper(), 0) >= LOG_LEVEL:
        first, *rest = args
        print(
            f"\n [{log_level.upper()}]: " + str(first),
            *rest,
            end="",
            file=sys.stderr,
            flush=True,
        )
