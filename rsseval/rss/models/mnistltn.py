# LTN architecture for MNIST
import torch
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.mnist_ltn_loss import ADDMNIST_SAT_AGG
import ltn


def get_parser() -> ArgumentParser:
    """Returns the argument parser for the current architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistLTN(CExt):
    """LTN architecture for MNIST"""

    NAME = "mnistltn"
    """
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    """

    def __init__(self, encoder, n_images=2, c_split=(), args=None):
        super(MnistLTN, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """

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
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.nr_classes = 3

        # opt and device
        self.task = args.task
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dictionary
        """
        # Image encoding
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])  # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

        # normalize concept preditions
        pCs = torch.nn.Softmax(dim=-1)(cs)

        if self.task == "addition":
            pred = torch.argmax(pCs[:, 0, :], dim=-1) + torch.argmax(
                pCs[:, 1, :], dim=-1
            )
            pred = F.one_hot(pred, 19)
        elif self.task == "product":
            pred = torch.argmax(pCs[:, 0, :], dim=-1) * torch.argmax(
                pCs[:, 1, :], dim=-1
            )
            pred = F.one_hot(pred, 82)
        elif self.task == "multiop":
            pred = (
                torch.argmax(pCs[:, 0, :], dim=-1) ** 2
                + torch.argmax(pCs[:, 1, :], dim=-1) ** 2
                + torch.argmax(pCs[:, 0, :], dim=-1)
                * torch.argmax(pCs[:, 1, :], dim=-1)
            )
            mask = pred > 14
            pred[mask] = torch.tensor(15, device=pred.device)
            pred = F.one_hot(pred, 16)

        return {"CS": cs, "YS": pred, "pCS": pCs}

    def get_loss(self, args):
        """Returns the loss function for the architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if loss is not implemented
        """
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            _and = None
            if args.and_op == "Godel":
                _and = ltn.fuzzy_ops.AndMin()
            elif args.and_op == "Prod":
                _and = ltn.fuzzy_ops.AndProd()
            else:
                _and = ltn.fuzzy_ops.AndLuk()

            And = ltn.Connective(_and)
            Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=args.p), quantifier="e")
            Forall = ltn.Quantifier(
                ltn.fuzzy_ops.AggregPMeanError(p=args.p), quantifier="f"
            )

            return ADDMNIST_SAT_AGG(And, Exists, Forall, args.task)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initializes the optimizer for this architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )
