# CBM model for BOIA
from utils.args import *
from models.utils.utils_problog import *
from models.utils.cbm_module import CBMModule
from utils.losses import SDDOIA_Cumulative
from utils.dpl_loss import SDDOIA_DPL
from utils.conf import get_device


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class SDDOIACBM(CBMModule):
    """CBM MODEL FOR SDDOIA"""

    NAME = "sddoiacbm"

    """
    SDDOIA
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
        super(SDDOIACBM, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )
        # device
        self.device = get_device()
        self.n_facts = n_facts
        self.nr_classes = nr_classes

        # how many images and explicit split of concepts
        self.c_split = c_split

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.n_facts, self.nr_classes), nn.Sigmoid()
        )

        # opt and device
        self.opt = None

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dict
        """
        # Image encoding
        cs = self.encoder(x)

        # expand concepts
        cs = cs.view(-1, cs.shape[1], 1)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)

        cs = torch.squeeze(cs, dim=-1)

        # CBM inference
        py = self.cbm_inference(cs)

        return {"CS": cs, "YS": py, "pCS": pCs}

    def cbm_inference(self, cs, query=None):
        """Performs CBM inference

        Args:
            self: instance
            cs: concepts logits
            query (default=None): query

        Returns:
            query_prob: query probability
        """

        # Pass the flattened input tensor through the classifier
        output = self.classifier(cs)

        query_prob = []

        # put the output in a way that it has probStopTrue, probStopFalse ...
        # going 2 by 2
        for i in range(self.nr_classes):
            query_prob.append(output[:, i])
            query_prob.append(1 - output[:, i])

        query_prob = torch.stack(query_prob, dim=1)

        # add a small offset
        query_prob = (query_prob + 1e-5) / (1 + 2 * 1e-5)

        return query_prob

    def normalize_concepts(self, concepts):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            concepts (torch.tensor): latents

        Returns:
            vec: normalized concepts
        """
        # Extract probs for each digit
        assert (
            len(concepts[concepts < 0]) == 0 and len(concepts[concepts > 1]) == 0
        ), concepts[:10, :, 0]

        pC = []
        for i in range(concepts.size(1)):
            # add offset
            c = torch.cat((1 - concepts[:, i], concepts[:, i]), dim=1) + 1e-5
            with torch.no_grad():
                Z = torch.sum(c, dim=1, keepdim=True)
            pC.append(c / Z)
        pC = torch.cat(pC, dim=1)

        return pC

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
        if args.dataset in ["sddoia", "boia", "presddoia"]:
            return SDDOIA_DPL(SDDOIA_Cumulative)
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
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )
