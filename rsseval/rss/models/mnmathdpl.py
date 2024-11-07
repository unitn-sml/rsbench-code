# DPL model for XOR
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import MNMATH_Cumulative
from utils.dpl_loss import MNMATH_DPL
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


class MNMATHDPL(DeepProblogModel):
    """DPL MODEL FOR XOR"""

    NAME = "mnmathdpl"

    """
    XOR but with synthetic data
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
        super(MNMATHDPL, self).__init__(
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
        self.n_images = 8

        # logic
        logic_sum = create_mnmath_sum(n_digits=10, sequence_len=4)
        logic_and = create_mnmath_prod(n_digits=10, sequence_len=4)
        logic_combine = create_mnist_and()
        
        # and, sum, combine
        self.logic_sum = logic_sum.to(self.device)
        self.logic_and = logic_and.to(self.device)
        self.combine = logic_combine.to(self.device)

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

        # cs = self.encoder(xs)
        # clen = len(cs[0].shape)

        for i in range(self.n_images):
            lc, _, _ = self.encoder(xs[i])  # sizes are ok
            cs.append(lc)
        clen = len(cs[0].shape)

        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)

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

        # Extract digit probability
        prob_digit1, prob_digit2, prob_digit3, prob_digit4 = pCs[:, 0, :], pCs[:, 1, :], pCs[:, 2, :], pCs[:, 3, :]

        # Extract again digit probability
        prob_digit5, prob_digit6, prob_digit7, prob_digit8 = pCs[:, 4, :], pCs[:, 5, :], pCs[:, 6, :], pCs[:, 7, :]

        # Compute worlds probability P(w) for sum
        probs_for_sum = outer_product(prob_digit1, prob_digit2, prob_digit3, prob_digit4)
        worlds_prob_sum = probs_for_sum.reshape(-1, int(self.n_facts ** (self.n_images / 2)))

        probs_for_prod = outer_product(prob_digit5, prob_digit6, prob_digit7, prob_digit8)
        worlds_prob_prod = probs_for_prod.reshape(-1, int(self.n_facts ** (self.n_images / 2)))

        # Compute query probability P(q)
        query_prob_sum = torch.zeros(
            size=(len(worlds_prob_sum), self.nr_classes), device=probs_for_sum.device
        )
        query_prob_prod = torch.zeros(
            size=(len(worlds_prob_prod), self.nr_classes), device=probs_for_prod.device
        )

        for i in range(self.nr_classes):
            query = i
            query_prob_sum[:, i] = self.compute_query_sum(query, worlds_prob_sum).view(-1)

        for i in range(self.nr_classes):
            query = i
            query_prob_prod[:, i] = self.compute_query_prod(query, worlds_prob_prod).view(-1)

        # add a small offset
        query_prob_prod += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob_prod, dim=-1, keepdim=True)
        query_prob_prod = query_prob_prod / Z

        # add a small offset
        query_prob_sum += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob_sum, dim=-1, keepdim=True)
        query_prob_sum = query_prob_sum / Z

        combined_tensor = torch.stack((query_prob_sum[:, 1], query_prob_prod[:, 1]), dim=1)
        
        return combined_tensor, None

    def compute_query_sum(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.logic_sum[:, query]

        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def compute_query_prod(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.logic_and[:, query]

        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def compute_query_combine(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.combine[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def normalize_concepts(self, z, split=8):
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
        if args.dataset in ["mnmath"]:
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
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    # override
    def to(self, device):
        super().to(device)
        self.logic_and = self.logic_and.to(device)
        self.logic_sum = self.logic_sum.to(device)
        self.combine = self.combine.to(device)
