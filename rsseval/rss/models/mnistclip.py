# Fully neural model for MNIST
import torch
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MNISTCLIP(nn.Module):
    """Fully neural MODEL FOR MNIST"""

    NAME = "mnistclip"
    """
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    """

    def __init__(
        self,
        encoder,
        n_images=2,
        c_split=(),
        args=None,
        n_facts=20,
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
        super(MNISTCLIP, self).__init__()
        # how many images and explicit split of concepts
        self.net = encoder
        self.n_facts = n_facts
        self.n_images = n_images
        self.joint = args.joint

        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.nr_classes = 3

        self.classifier = nn.Sequential(
            nn.Linear(
                self.n_facts * self.n_images,
                self.n_facts * self.n_images,
                dtype=torch.float64,
            ),
            nn.ReLU(),
            nn.Linear(
                self.n_facts * self.n_images,
                self.n_facts * self.n_images,
                dtype=torch.float64,
            ),
            nn.ReLU(),
            nn.Linear(
                self.n_facts * self.n_images, self.nr_classes, dtype=torch.float64
            ),
            nn.Softmax(dim=1),
        )

        # opt and device
        self.opt = None
        self.device = get_device()

    def cmb_inference(self, cs, query=None):
        """Performs inference inference

        Args:
            self: instance
            cs: concepts logits
            query (default=None): query

        Returns:
            query_prob: query probability
        """

        # flatten the cs
        flattened_cs = cs.view(cs.shape[0], cs.shape[1] * cs.shape[2])

        # Pass the flattened input tensor through the classifier
        query_prob = self.classifier(flattened_cs)

        # add a small offset
        query_prob = query_prob + 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob, dim=-1, keepdim=True)
        query_prob = query_prob / Z

        return query_prob

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        cs = torch.zeros((x.shape[0], self.n_images, self.n_facts)).to(x.device)
        cs[:] = -1  # set at -1, they are not computed

        pCs = torch.zeros((x.shape[0], self.n_images, self.n_facts)).to(x.device)
        pCs[:] = 1.0 / self.n_facts  # set at uniform, they are not computed

        py = self.cmb_inference(x)  # cs

        return {"CS": cs, "YS": py, "pCS": pCs}

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
        if args.dataset in [
            "addmnist",
            "shortmnist",
            "restrictedmnist",
            "halfmnist",
            "clipshortmnist",
        ]:
            return ADDMNIST_DPL(ADDMNIST_Cumulative)
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
        self.opt = torch.optim.Adam(self.parameters(), args.lr)
