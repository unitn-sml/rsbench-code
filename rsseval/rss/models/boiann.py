# Fully neural model for BOIA
from utils.args import *
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import SDDOIA_DPL
from models.sddoiann import SDDOIAnn


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class BOIAnn(SDDOIAnn):
    """Fully neural MODEL FOR BOIA"""

    NAME = "boiann"
    """
    BOIA
    """

    def __init__(
        self,
        encoder,
        n_images=2,
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
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=20): number of concepts
            nr_classes (int, nr_classes): number of classes

        Returns:
            None: This function does not return a value.
        """
        super(BOIAnn, self).__init__(
            encoder,
            n_images=2,
            c_split=(),
            args=None,
            model_dict=None,
            n_facts=21,
            nr_classes=4,
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
        if args.dataset in ["boia", "clipboia"]:
            return SDDOIA_DPL(SDDOIA_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")
