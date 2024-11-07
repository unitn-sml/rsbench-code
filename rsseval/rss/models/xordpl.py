# DPL model for XOR
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import XOR_Cumulative
from utils.dpl_loss import XOR_DPL
from models.utils.ops import outer_product


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class XORDPL(DeepProblogModel):
    """DPL MODEL FOR XOR"""

    NAME = "xordpl"

    """
    XOR but with synthetic data
    """

    def __init__(
        self,
        encoder,
        n_images=4,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=2,
        nr_classes=2,
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
        super(XORDPL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )
        # device
        self.device = get_device()
        self.n_facts = n_facts

        # how many images and explicit split of concepts
        self.c_split = c_split
        self.args = args
        self.n_images = n_images

        # logic
        logic = create_xor(n_digits=2, sequence_len=4)
        self.xor = logic.to(self.device)

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
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        
        # cs
        cs = self.encoder(xs)

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)

        # Problog inference to compute worlds and query probability distributions
        py, worlds_prob = self.problog_inference(pCs)

        return {"CS": cs, "YS": py, "pCS": pCs}

    def problog_inference(self, pCs, query=None):
        """Performs ProbLog inference to retrieve the worlds probability distribution P(w).
        Works with an arbitrary number of encoded bits (digits).

        Args:
            self: instance
            pCs: probability of concepts (shape: [batch_size, num_digits, num_classes])
            query (default=None): query

        Returns:
            query_prob: query probability
            worlds_prob: worlds probability
        """

        
        # Extract first and second digit probability
        prob_digit1, prob_digit2, prob_digit3, prob_digit4 = pCs[:, 0, :], pCs[:, 1, :], pCs[:, 2, :], pCs[:, 3, :]

        # Compute worlds probability P(w) (the two digits values are independent)
        probs = outer_product(prob_digit1, prob_digit2, prob_digit3, prob_digit4)
        worlds_prob = probs.reshape(-1, self.n_facts ** self.n_images)

        # Compute query probability P(q)
        query_prob = torch.zeros(
            size=(len(probs), self.nr_classes), device=probs.device
        )

        for i in range(self.nr_classes):
            query = i
            query_prob[:, i] = self.compute_query(query, worlds_prob).view(-1)

        # add a small offset
        query_prob += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob, dim=-1, keepdim=True)
        query_prob = query_prob / Z

        return query_prob, worlds_prob

    def compute_query(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.xor[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def normalize_concepts(self, z, split=4):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): number of splits (number of digits)

        Returns:
            vec: normalized concepts
        """
        # List to hold normalized probabilities for each digit
        normalized_probs = []
        
        # Small value to avoid underflow
        eps = 1e-5

        # Iterate over each digit's latent vector
        for i in range(split):
            # Extract the probability for the current digit
            prob_digit = z[:, i, :]

            # Apply softmax to ensure the probabilities sum to 1
            prob_digit = nn.Softmax(dim=1)(prob_digit)
            
            # Add a small epsilon to avoid ProbLog underflow
            prob_digit = prob_digit + eps
            
            # Normalize the probabilities
            with torch.no_grad():
                Z = torch.sum(prob_digit, dim=-1, keepdim=True)
            prob_digit = prob_digit / Z  # Normalization
            
            # Append the normalized probability to the list
            normalized_probs.append(prob_digit)

        # Stack the normalized probabilities along the dimension for digits
        normalized_probs = torch.stack(normalized_probs, dim=1)
        return normalized_probs

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
        if args.dataset in ["xor"]:
            return XOR_DPL(XOR_Cumulative)
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

    # override
    def to(self, device):
        super().to(device)
        self.xor = self.xor.to(device)
