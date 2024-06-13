# MNIST PCBM with DPL architecture
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from models.mnistdpl import MnistDPL
from utils.args import *
from utils.conf import get_device
from models.utils.deepproblog_modules import GraphSemiring
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import ADDMNIST_DPL


def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistPcbmDPL(MnistDPL):
    """DPL with PCBM encoder for MNIST"""

    NAME = "mnistpcbmdpl"
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
            encoder (nn.Module): encoder network
            n_images (int, default=2): number of images
            c_split: concept split
            args (default=None): command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=20): number of concepts
            nr_classes (int, default=19): number of classes

        Returns:
            None: This function does not return a value.
        """
        super(MnistPcbmDPL, self).__init__(
            encoder, n_images, c_split, args, model_dict, n_facts, nr_classes
        )

        self.positives = torch.nn.Parameter(
            torch.randn(1, int(self.n_facts), 16), requires_grad=True
        )
        self.negatives = torch.nn.Parameter(
            torch.randn(1, int(self.n_facts), 16), requires_grad=True
        )

        self.negative_scale = 1 * torch.ones(1, device=self.device)
        self.shift = torch.ones(0, device=self.device)

    def batchwise_cdist(self, samples1, samples2, eps=1e-6):
        """Batchwise distance, adaptd from the original PCBM repository

        Args:
            self: instance
            samples1: first sample
            samples2: second sample
            eps (float, default=1e-6): epsilon value

        Returns:
            dist: batchwise distance
        """
        if len(samples1.size()) not in [3, 4, 5] or len(samples2.size()) not in [
            3,
            4,
            5,
        ]:
            raise RuntimeError(
                "expected: 4-dim tensors, got: {}, {}".format(
                    samples1.size(), samples2.size()
                )
            )

        if samples1.size(0) == samples2.size(0):
            batch_size = samples1.size(0)
        elif samples1.size(0) == 1:
            batch_size = samples2.size(0)
        elif samples2.size(0) == 1:
            batch_size = samples1.size(0)
        elif samples1.shape[1] == samples2.shape[1]:
            samples1 = samples1.unsqueeze(2)
            samples2 = samples2.unsqueeze(3)
            samples1 = samples1.unsqueeze(1)
            samples2 = samples2.unsqueeze(0)
            result = torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
            return result.view(*result.shape[:-2], -1)
        else:
            raise RuntimeError(
                f"samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities "
                "are non-broadcastable."
            )

    def compute_distance(
        self, pred_embeddings, z_tot, negative_scale=None, shift=None, reduction="mean"
    ):
        """Compute distances between predicted embeddings and latents z

        Args:
            self: instance
            pred_embeddings: predicted embeddings
            z tot: latents
            negative_scale (default=None): negative scale
            shift (default=None): negative scale
            reduction (str, default=None): which reduction to use

        Returns:
            dist: mean batchwise distance
            probability: mean probability
        """
        negative_scale = (
            self.negative_scale if negative_scale is None else negative_scale
        )

        distance = self.batchwise_cdist(pred_embeddings, z_tot)

        distance = distance.permute(0, 2, 3, 1)

        logits = -negative_scale.view(1, -1, 1, 1) * distance
        prob = torch.nn.functional.softmax(logits, dim=-1)
        if reduction == "none":
            return logits, prob
        return logits.mean(axis=-2), prob.mean(axis=-2)

    def forward(self, x):
        """Forward pass

        Args:
            self: instance
            x (torch.tensor): input x

        Returns:
            out_dict: output dictionary
        """
        B = x.shape[0]

        # Image encoding
        mus, logvars = [], []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            _, mu, logvar = self.encoder(xs[i])  # sizes are ok
            # mu.view(B, 5, -1)
            # logvar.view(B, 5, -1)
            mus.append(mu), logvars.append(logvar)

        clen = len(mus[0].shape)

        # mus = torch.stack(mus, dim=1) if clen == 2 else torch.cat(mus, dim=1)
        # logvars = torch.stack(logvars, dim=1) if clen == 2 else torch.cat(logvars, dim=1)

        # print("Mus", mus.shape)
        # print("Logvars", logvars.shape)

        z_plus = F.normalize(self.positives, p=2, dim=-1)
        z_nega = F.normalize(self.negatives, p=2, dim=-1)

        # print("Z plus", z_plus.shape)
        # print("Z neg", z_nega.shape)

        z_tot = torch.cat([z_nega, z_plus]).unsqueeze(-2)
        # z_tot = torch.unsqueeze(z_tot, dim=0)
        # print("Z tot", z_tot.shape)

        # sampling
        prob_Cs, c_logits = [], []
        for i in range(self.n_images):

            latents = F.normalize(mus[i], p=2, dim=-1)
            logsigma = torch.clip(logvars[i], max=10)

            #  [batch, n_concepts, n_samples, latent_dim]
            pred_embeddings = sample_gaussian_tensors(latents, logsigma, 10)

            # print("Pred embeddings", pred_embeddings.shape)

            concept_logit, concept_prob = self.compute_distance(pred_embeddings, z_tot)
            # print("concept_logit", concept_logit.shape)
            # print("concept_prob", concept_prob.shape)

            prob_Cs.append(concept_prob[..., 1].unsqueeze(1))
            c_logits.append(concept_logit[..., 1].unsqueeze(1))

        prob_Cs = torch.cat(prob_Cs, dim=1)
        c_logits = torch.cat(c_logits, dim=1)

        # print(concept_probs.shape)
        # quit()

        # normalize concept preditions
        pCs = self.normalize_concepts(c_logits)

        # Problog inference to compute worlds and query probability distributions
        py, worlds_prob = self.problog_inference(pCs)

        return {"CS": pCs, "YS": py, "pCS": pCs, "MUS": mus, "LOGVARS": logvars}

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance,
            z: latents
            splits (int, default=2): number of splits

        Returns:
            vec: normalized concept vector
        """
        # Extract probs for each digit

        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        # # Add stochasticity on prediction
        # prob_digit1 += 0.5 * torch.randn_like(prob_digit1, device=prob_digit1.device)
        # prob_digit2 += 0.5 * torch.randn_like(prob_digit2, device=prob_digit1.device)

        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid ProbLog underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization

        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)

    @staticmethod
    def get_loss(args):
        """Returns the loss function for this architecture

        Args:
            args: command line arguments

        Returns:
            loss function: loss function for this architecture

        Raises:
            err: NotImplementedError if the loss function is not present
        """
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            return ADDMNIST_DPL(ADDMNIST_Cumulative, pcbm=True)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initializes the optimizer

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            [*self.parameters(), self.positives, self.negatives], args.lr
        )


def sample_gaussian_tensors(mu, logsigma, num_samples):
    """Sample gaussian tensor from a Gaussian distribution parametrized with mu and logsigma

    Args:
        mu: mean of the Gaussian
        logsigma: std of the Gaussian
        num_samples (int): number of samples to sample

    Returns:
        samples: samples from the distribution
    """
    eps = torch.randn(
        mu.size(0),
        mu.size(1),
        num_samples,
        mu.size(2),
        dtype=mu.dtype,
        device=mu.device,
    )
    samples_sigma = eps.mul(torch.exp(logsigma.unsqueeze(2) * 0.5))
    samples = samples_sigma.add_(mu.unsqueeze(2))
    return samples
