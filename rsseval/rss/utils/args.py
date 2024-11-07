# Args module

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # dataset
    parser.add_argument(
        "--dataset",
        default="addmnist",
        type=str,
        choices=DATASET_NAMES,
        help="Which dataset to perform experiments on.",
    )
    parser.add_argument(
        "--task",
        default="addition",
        type=str,
        choices=[
            "addition",
            "product",
            "multiop",
            "base",
            "red_triangle",
            "triangle_circle",
            "patterns",
            "mini_patterns",
            "boia",
            "xor",
            "mnmath"
        ],
        help="Which operation to choose.",
    )
    # model settings
    parser.add_argument(
        "--model",
        type=str,
        default="mnistdpl",
        help="Model name.",
        choices=get_all_models(),
    )
    parser.add_argument(
        "--c_sup",
        type=float,
        default=0,
        help="Fraction of concept supervision on concepts",
    )
    parser.add_argument(
        "--c_sup_ltn",
        type=int,
        default=0,
        help="Whether to use concept supervision when training LTN model (particularly on Kandinsky)",
    )
    parser.add_argument(
        "--which_c",
        type=int,
        nargs="+",
        default=[-1],
        help="Which concepts explicitly supervise (-1 means all)",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        default=False,
        help="Process the image as a whole.",
    )
    parser.add_argument(
        "--splitted",
        action="store_true",
        default=False,
        help="Create different encoders.",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        default=False,
        help="Activate entropy on batch.",
    )
    # weights of logic
    parser.add_argument(
        "--w_sl", type=float, default=10, help="Weight of Semantic Loss"
    )
    # LTN semantics for logical operators
    parser.add_argument(
        "--and_op", type=str, default="Prod", help="Semantic for the And Operator"
    )
    parser.add_argument(
        "--or_op", type=str, default="Prod", help="Semantic for the Or Operator"
    )
    parser.add_argument(
        "--imp_op", type=str, default="Prod", help="Semantic for the Implies Operator"
    )
    parser.add_argument(
        "--p", type=int, default="2", help="Hyper-parameter for LTN quantifiers grade"
    )
    # weight of mitigation
    parser.add_argument("--gamma", type=float, default=1, help="Weight of mitigation")
    # additional hyperparams
    parser.add_argument(
        "--w_rec", type=float, default=1, help="Weight of Reconstruction"
    )
    parser.add_argument("--beta", type=float, default=2, help="Multiplier of KL")
    parser.add_argument("--w_h", type=float, default=1, help="Weight of entropy")
    parser.add_argument("--w_c", type=float, default=1, help="Weight of concept sup")

    # optimization params
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)."
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup epochs.")
    parser.add_argument(
        "--exp_decay", type=float, default=1.0, help="Exp decay of learning rate."
    )

    # learning hyperams
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Number of epochs per task."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

    # deep ensembles
    parser.add_argument(
        "--boia-model",
        default="ce",
        choices=["ce", "bce"],
        type=str,
        help="Training using CE or BCE",
    )

    parser.add_argument(
        "--backbone",
        default="conceptizer",
        choices=["conceptizer", "neural"],
        type=str,
        help="Which backbone to use",
    )
    # parameters for tuning
    parser.add_argument(
        "--count",
        default=30,
        type=int,
        help="Number of hyper-params configurations that has to be tried during tuning.",
    )
    parser.add_argument(
        "--val_metric",
        default="accuracy",
        choices=["accuracy", "f1"],
        type=str,
        help="Validation metric that has to be minimized during hyper-parameter tuning.",
    )
    parser.add_argument(
        "--tuning",
        default=False,
        action="store_true",
        help="Whether to perform tuning of the specified model.",
    )
    parser.add_argument(
        "--proj_name",
        default="",
        type=str,
        help="Weights and Biases project name where the runs have to be logged.",
    )
    parser.add_argument(
        "--entity",
        default="",
        type=str,
        help="Weights and Biases project entity (username).",
    )
    parser.add_argument(
        "--boia-ood-knowledge",
        default=False,
        action="store_true",
        help="Whether to employ BOIA OOD-knowledge (Ambulance) only for DPL",
    )


def add_management_args(parser: ArgumentParser) -> None:
    """Adds the arguments used in management

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # random seed
    parser.add_argument("--seed", type=int, default=None, help="The random seed.")
    # verbosity
    parser.add_argument("--notes", type=str, default=None, help="Notes for this run.")
    parser.add_argument("--non_verbose", action="store_true")
    # logging
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Enable wandb logging -- set name of project",
    )
    # checkpoints
    parser.add_argument(
        "--checkin",
        type=str,
        default=None,
        help="location and path FROM where to load ckpt.",
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        default=False,
        help="save the model to data/ckpts.",
    )
    # post-hoc evaluation
    parser.add_argument(
        "--posthoc",
        action="store_true",
        default=False,
        help="Used to evaluate only the loaded model",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        default=False,
        help="Used to non-linear probe the model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Used to evaluate on the validation set for hyperparameters search",
    )
    # preprocessing option
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Used to preprocess dataset",
    )


def add_test_args(parser: ArgumentParser) -> None:
    """Arguments for the Test part of the code

    Args:
        parser: the parser instance

    Returns:
        None: This function does not return a value.
    """
    # random seed
    parser.add_argument(
        "--use_ood", action="store_true", help="Use Out of Distribution test samples."
    )
