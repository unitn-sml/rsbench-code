# config module

import random
import torch
import numpy as np
import os


def get_device() -> torch.device:
    """Returns the GPU device if available else CPU.

    Returns:
        device: device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """Returns the base bath where to log accuracies and tensorboard data.

    Returns:
        base_path (str): base path
    """
    return "./data/"


def set_random_seed(seed: int) -> None:
    """Sets the seeds at a certain value.

    Args:
        param seed: the value to be set

    Returns:
        None: This function does not return a value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_path(path) -> None:
    """Create path function, create folder if it does not exists

    Args:
        path (str): path value

    Returns:
        None: This function does not return a value.
    """
    if not os.path.exists(path):
        os.makedirs(path)
