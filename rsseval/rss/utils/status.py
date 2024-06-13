# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
import sys
import os
from utils.conf import base_path
from typing import Any, Dict, Union
from torch import nn
from argparse import Namespace


def create_stash(model: nn.Module, args: Namespace, dataset) -> Dict[Any, str]:
    """Creates the dictionary where to save the model status.

    Args:
        model: the model
        args: the current arguments
        dataset: the dataset at hand

    Returns:
        model_stash: dictionary stash
    """
    now = datetime.now()
    model_stash = {"task_idx": 0, "epoch_idx": 0, "batch_idx": 0}
    name_parts = [args.dataset, model.NAME]
    if "buffer_size" in vars(args).keys():
        name_parts.append("buf_" + str(args.buffer_size))
    name_parts.append(now.strftime("%Y%m%d_%H%M%S_%f"))
    model_stash["model_name"] = "/".join(name_parts)
    model_stash["mean_accs"] = []
    model_stash["args"] = args
    model_stash["backup_folder"] = os.path.join(
        base_path(), "backups", dataset.SETTING, model_stash["model_name"]
    )
    return model_stash


def create_fake_stash(model: nn.Module, args: Namespace) -> Dict[Any, str]:
    """Create a fake stash, containing just the model name. This is used in general continual, as it is useless to backup a lightweight MNIST-360 training.

    Args:
        model: the model
        args: the arguments of the call

    Returns
        value: a dict containing a fake stash
    """
    now = datetime.now()
    model_stash = {"task_idx": 0, "epoch_idx": 0}
    name_parts = [args.dataset, model.NAME]
    if "buffer_size" in vars(args).keys():
        name_parts.append("buf_" + str(args.buffer_size))
    name_parts.append(now.strftime("%Y%m%d_%H%M%S_%f"))
    model_stash["model_name"] = "/".join(name_parts)

    return model_stash


def progress_bar(i: int, max_iter: int, epoch: Union[int, str], loss: float) -> None:
    """Prints out the progress bar on the stderr file.

    Args:
        i: the current iteration
        max_iter: the maximum number of iteration
        epoch: the epoch
        task_number: the task index
        loss: the current value of the loss function

    Returns:
        None: This function does not return a value.
    """
    # if not (i + 1) % 10 or (i + 1) == max_iter:
    progress = min(float((i + 1) / max_iter), 1)
    progress_bar = ("█" * int(50 * progress)) + ("┈" * (50 - int(50 * progress)))
    print(
        "\r[ {} ] epoch {}: |{}| loss: {}".format(
            datetime.now().strftime("%m-%d | %H:%M"),
            epoch,
            progress_bar,
            round(loss, 8),
        ),
        file=sys.stderr,
        end="",
        flush=True,
    )
