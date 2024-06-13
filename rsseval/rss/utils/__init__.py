# init utils module
import os
import builtins
import sys


def create_if_not_exists(path: str) -> None:
    """Creates the specified folder if it does not exist.

    Args:
        path: the complete path of the folder to be created

    Returns:
        None: This function does not return a value.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def fprint(*args, **kwargs):
    """Flushing print

    Args:
        args: arguments
        kwargs: key-value arguments

    Returns:
        None: This function does not return a value.
    """
    builtins.print(*args, **kwargs)
    sys.stdout.flush()
