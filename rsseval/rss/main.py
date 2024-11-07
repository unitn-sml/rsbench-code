# This is the main module
# It provides an overview of the program purpose and functionality.

import sys, os
import torch
import argparse
import importlib
import setproctitle, socket, uuid
import datetime

from datasets import get_dataset
from models import get_model
from utils.train import train
from utils.test import test
from utils.preprocess_resnet import preprocess
from utils.conf import *
import signal
from utils.args import *
from utils.checkpoint import save_model, create_load_ckpt
from utils.probe import probe

from argparse import Namespace
import wandb

conf_path = os.getcwd() + "."
sys.path.append(conf_path)


class TerminationError(Exception):
    """Error raised when a termination signal is received"""

    def __init__(self):
        """Init method

        Args:
            self: instance

        Returns:
            None: This function does not return a value.
        """
        super().__init__("External signal received: forcing termination")


def __handle_signal(signum: int, frame):
    """For program termination on cluster

    Args:
        signum (int): signal number
        frame: frame

    Returns:
        None: This function does not return a value.

    Raises:
        TerminationError: Always.
    """
    raise TerminationError()


def register_termination_handlers():
    """Makes this process catch SIGINT and SIGTERM. When the process receives such a signal after this call, a TerminationError is raised.

    Returns:
        None: This function does not return a value.
    """

    signal.signal(signal.SIGINT, __handle_signal)
    signal.signal(signal.SIGTERM, __handle_signal)


def parse_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Reasoning Shortcut", allow_abbrev=False
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cext",
        help="Model for inference.",
        choices=get_all_models(),
    )
    parser.add_argument(
        "--load_best_args",
        action="store_true",
        help="Loads the best arguments for each method, " "dataset and memory buffer.",
    )

    torch.set_num_threads(4)

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module("models." + args.model)

    # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    get_parser = getattr(mod, "get_parser")
    parser = get_parser()
    parser.add_argument(
        "--project", type=str, default="Reasoning-Shortcuts", help="wandb project"
    )
    add_test_args(parser)
    args = parser.parse_args()  # this is the return

    # load args related to seed etc.
    set_random_seed(args.seed) if args.seed is not None else set_random_seed(42)

    return args


def tune(args):
    """
    This function performs a hyper-parameter tuning of the model using a WandB sweep.

    Args:
        args: parsed command line arguments
    """
    sweep_conf = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": args.val_metric},
        "parameters": {
            "batch_size": {"values": [32, 64, 128, 256, 512]},
            "lr": {"values": [0.0001, 0.001, 0.01]},
            "weight_decay": {"values": [0.0, 0.0001, 0.001, 0.01, 0.1]},
        },
    }

    if "ltn" in args.model:
        sweep_conf["parameters"]["p"] = {"values": [2, 4, 6, 8, 10]}
        sweep_conf["parameters"]["and_op"] = {"values": ["Godel", "Prod"]}
        sweep_conf["parameters"]["or_op"] = {"values": ["Godel", "Prod"]}
        sweep_conf["parameters"]["imp_op"] = {"values": ["Godel", "Prod"]}

    if args.c_sup > 0:
        sweep_conf["parameters"]["w_c"] = {"values": [1, 2, 5]}

    if args.entropy > 0:
        sweep_conf["parameters"]["w_h"] = {"values": [1, 2, 5, 8, 10]}

    def train_conf():
        with wandb.init(project=args.proj_name, config=sweep_conf, entity=args.entity):
            config = wandb.config
            args.batch_size = config.batch_size
            args.lr = config.lr
            args.weight_decay = config.weight_decay
            if "ltn" in args.model:
                args.p = config.p
                args.and_op = config.and_op
                args.or_op = config.or_op
                args.imp_op = config.imp_op
            dataset = get_dataset(args)

            # Load dataset, model, loss, and optimizer
            encoder, decoder = dataset.get_backbone()
            n_images, c_split = dataset.get_split()
            model = get_model(args, encoder, decoder, n_images, c_split)
            loss = model.get_loss(args)
            model.start_optim(args)

            train(model, dataset, loss, args)

    sweep_id = wandb.sweep(sweep=sweep_conf, project=args.proj_name)
    wandb.agent(sweep_id, function=train_conf, count=args.count)


def main(args):
    """Main function. Provides functionalities for training, testing and active learning.

    Args:
        args: parsed command line arguments.

    Returns:
        None: This function does not return a value.
    """
    if not args.tuning:
        # Add uuid, timestamp and hostname for logging
        args.conf_jobnum = str(uuid.uuid4())
        args.conf_timestamp = str(datetime.datetime.now())
        args.conf_host = socket.gethostname()
        dataset = get_dataset(args)

        # Load dataset, model, loss, and optimizer
        encoder, decoder = dataset.get_backbone()
        n_images, c_split = dataset.get_split()
        model = get_model(args, encoder, decoder, n_images, c_split)
        loss = model.get_loss(args)
        model.start_optim(args)

        # SAVE A BASE MODEL OR LOAD IT, LOAD A CHECKPOINT IF PROVIDED
        # model = create_load_ckpt(model, args)

        # set job name
        setproctitle.setproctitle(
            "{}_{}_{}".format(
                args.model,
                args.buffer_size if "buffer_size" in args else 0,
                args.dataset,
            )
        )

        # perform posthoc evaluation/ cl training/ joint training
        print("    Chosen device:", model.device)

        if args.preprocess:
            preprocess(model, dataset, args)
            print("\n ### Closing ###")
            quit()

        if args.probe:
            probe(model, dataset, args)
        elif args.posthoc:
            test(model, dataset, args)  # test the model if post-hoc is passed
        else:
            train(model, dataset, loss, args)  # train the model otherwise
            save_model(model, args)  # save the model parameters
    else:
        tune(args)

    print("\n ### Closing ###")


if __name__ == "__main__":
    args = parse_args()

    main(args)
