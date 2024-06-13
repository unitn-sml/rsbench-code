# Kandinksy DPL
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import KAND_DPL
from models.utils.ops import outer_product
from models.utils.cbm_module import CBMModule


def get_parser() -> ArgumentParser:
    """Argument parser for Kandinsky DPL

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class KandCBM(CBMModule):
    """Kandinsky DPL model"""

    NAME = "kandcbm"
    """
    Kandinsky CBM
    """

    def __init__(
        self,
        encoder,
        n_images=3,
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
        super(KandCBM, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # Worlds-queries matrix
        self.n_facts = 6
        self.n_predicates = 9
        self.nr_classes = 1

        # opt and device
        self.opt = None
        self.device = get_device()

        # the two bottlenecks
        self.first_bottleneck = nn.Sequential(
            nn.Linear(self.n_facts * 3, self.n_predicates), nn.Sigmoid()
        )
        self.second_bottleneck = nn.Sequential(
            nn.Linear(self.n_predicates * 3, self.nr_classes), nn.Sigmoid()
        )

    def forward(self, x, activate_simple_concepts=False):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector
            activate_simple_concepts (bool, default=False): whether to return concepts in a simple manner

        Returns:
            c: simple concepts
            out_dict: output dictionary
        """

        # Image encoding
        cs, pCs, preds = [], [], []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            lc, _ = self.encoder(xs[i])  # sizes are ok
            pc = self.normalize_concepts(lc)
            pred = self.cbm_inference_1(lc)
            cs.append(lc), pCs.append(pc), preds.append(pred)

        clen = len(cs[0].shape)

        cs = torch.stack(cs, dim=1) if clen > 1 else torch.cat(cs, dim=1)
        pCs = torch.stack(pCs, dim=1) if clen > 1 else torch.cat(pCs, dim=1)

        preds = torch.stack(preds, dim=1) if clen > 1 else torch.cat(preds, dim=1)

        py = self.cbm_inference_2(preds)

        return {"CS": cs, "YS": py, "pCS": pCs, "PREDS": preds}

    def cbm_inference_1(self, cs, query=None):
        """CBM inference

        Args:
            self: instance
            pCs: concepts
            preds: predictions

        Returns:
            query_prob: query probabilities
        """
        return self.first_bottleneck(cs)

    def cbm_inference_2(self, cs, query=None):
        """CBM inference

        Args:
            self: instance
            pCs: concepts
            preds: predictions

        Returns:
            query_prob: query probabilities
        """
        tmp = torch.flatten(cs, start_dim=1)
        out = self.second_bottleneck(tmp)
        complement = 1 - out
        new_out = torch.concatenate((out, complement), dim=1)
        return new_out

    def normalize_concepts(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents

        Returns:
            norm_concepts: normalized concepts
        """

        def soft_clamp(h, dim=-1):
            h = nn.Softmax(dim=dim)(h)

            eps = 1e-5
            h = h + eps
            with torch.no_grad():
                Z = torch.sum(h, dim=dim, keepdim=True)
            h = h / Z
            return h

        # TODO: the 3 here is hardcoded, relax to arbitrary concept encodings?
        pCi = torch.split(z, 3, dim=-1)  # [batch_size, 24] -> [8, batch_size, 3]

        norm_concepts = torch.cat(
            [soft_clamp(c) for c in pCi], dim=-1
        )  # [8, batch_size, 3] -> [batch_size, 24]

        return norm_concepts

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
        if args.dataset in ["kandinsky", "prekandinsky", "minikandinsky"]:
            return KAND_DPL(KAND_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initialize optimizer

        Args:
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(),
            args.lr,
            weight_decay=1e-5,
        )
