# Fully neural model for Kandinsky
import torch
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import MNMATH_DPL


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MNMATHnn(nn.Module):
    """Fully neural MODEL FOR MNMATH"""

    NAME = "mnmathnn"
    """
    XOR
    """

    def __init__(
        self,
        encoder,
        n_images=8,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=10,
        nr_classes=2,
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
        super(MNMATHnn, self).__init__()
        # how many images and explicit split of concepts
        self.net = encoder
        self.n_facts = n_facts
        self.n_images = 8
        self.joint = args.joint

        # opt and device
        self.opt = None
        self.n_predicates = 2
        self.device = get_device()


    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """

        # split the images
        cs = torch.zeros((x.shape[0], self.n_images, self.n_facts)).to(
            x.device
        )

        cs[:] = -1  # set at -1, they are not computed

        preds = torch.zeros((x.shape[0], self.n_images, self.n_predicates)).to(x.device)
        preds[:] = -1  # set at -1, they are not computed

        pCs = torch.zeros((x.shape[0], self.n_images, self.n_facts)).to(
            x.device
        )
        pCs[:] = 1.0 / self.n_facts  # set at uniform, they are not computed

        py = self.net(x)

        return {"CS": cs, "YS": py, "pCS": pCs, "PREDS": preds}

    @staticmethod
    def get_loss(args):
        """Loss function for KandDPL

        Args:
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError
        """
        if args.dataset in [
            "mnmath",
        ]:
            return MNMATH_DPL(MNMATH_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initialize optimizer

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(self.net.parameters(), args.lr)
