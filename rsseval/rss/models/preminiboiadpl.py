# DPL model for Preprocessed MINIBOIA
from utils.args import *
from models.utils.utils_problog import *
from models.miniboiadpl import MiniBoiaDPL
from utils.losses import MINIBOIA_Cumulative
from utils.dpl_loss import MINIBOIA_DPL


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class PreMiniBoiaDPL(MiniBoiaDPL):
    """DPL MODEL FOR PreMINIBOIA"""

    NAME = "preminiboiadpl"

    """
    PreMINIBOIA
    """

    def __init__(
        self,
        encoder,
        n_images=1,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=21,
        nr_classes=4,
    ):
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=1): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=21): number of concepts
            nr_classes (int, nr_classes): number of classes for the multiclass classification problem
            retun_embeddings (bool): whether to return embeddings

        Returns:
            None: This function does not return a value.
        """
        super(PreMiniBoiaDPL, self).__init__(
            encoder,
            n_images=n_images,
            c_split=c_split,
            args=args,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

    @staticmethod
    def get_loss(args):
        """Loss function for the architecture

        Args:
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if the loss function is not available
        """
        if args.dataset in ["preminiboia"]:
            return MINIBOIA_DPL(MINIBOIA_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")
