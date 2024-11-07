# DPL model for BOIA
from utils.args import *
from models.utils.utils_problog import *
from models.sddoiadpl import SDDOIADPL
from utils.losses import SDDOIA_Cumulative
from utils.dpl_loss import SDDOIA_DPL


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class BoiaDPL(SDDOIADPL):
    """DPL MODEL FOR BOIA"""

    NAME = "boiadpl"

    """
    BOIA
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
        super(BoiaDPL, self).__init__(
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
        if args.dataset in ["boia"]:
            return SDDOIA_DPL(SDDOIA_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")
