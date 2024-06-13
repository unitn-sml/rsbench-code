import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *


def get_parser() -> ArgumentParser:
    """Argument parser for preprocessed Kandinsky

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class KANDPreProcess(DeepProblogModel):
    """Kandinsky preprocessed DPL model"""

    NAME = "kandpreprocess"
    """
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    """

    def __init__(
        self,
        encoder,
        n_images=2,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=20,
        nr_classes=19,
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
        super(KANDPreProcess, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            emb: dictionary of embeddings
        """
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc = self.encoder(xs[i])  # sizes are ok

            cs.append(lc)

        clen = len(cs[0].shape)

        embs = torch.stack(cs, dim=1) if clen > 1 else torch.cat(cs, dim=1)

        return {"EMBS": embs}

    @staticmethod
    def get_loss(args=None):
        """Loss function, not implemented

        Args:
            args (default=None): command line arguments

        Returns:
            None: This function does not return a value.
        """
        return None

    def start_optim(self, args):
        """Start optimizer, not implemented

        Args:
            self: instance
            args (default=None): command line arguments

        Returns:
            None: This function does not return a value.
        """
        return
