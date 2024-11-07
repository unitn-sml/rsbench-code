# DPL model for BOIA
import torch
import torch.nn as nn
from utils.args import *
from models.boialtn import BOIALTN
from utils.conf import get_device
from utils.losses import *
from utils.boia_ltn_loss import SDDOIA_SAT_AGG
import ltn


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class SDDOIALTN(BOIALTN):
    """DPL MODEL FOR BOIA"""

    NAME = "sddoialtn"

    """
    SDDOIA
    """

    def __init__(self, encoder, n_images=2, c_split=(), args=None):
        super(SDDOIALTN, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split, args=args
        )
