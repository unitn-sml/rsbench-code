import torch
from models.mnistltn import MnistLTN
from utils.args import *
from utils.conf import get_device
from models.utils.deepproblog_modules import GraphSemiring
from models.utils.utils_problog import *
from utils.losses import *
from utils.mnist_ltn_loss import ADDMNIST_SAT_AGG


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MnistPcbmLTN(MnistLTN):
    NAME = "mnistpcbmltn"
    """
    MNIST OPERATIONS AMONG TWO DIGITS. IT WORKS ONLY IN THIS CONFIGURATION.
    """

    def __init__(self, encoder, n_images=2, c_split=(), args=None):
        super(MnistPcbmLTN, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split, args=args
        )

        self.positives = torch.nn.Parameter(
            torch.randn(1, int(self.n_facts), 16), requires_grad=True
        )
        self.negatives = torch.nn.Parameter(
            torch.randn(1, int(self.n_facts), 16), requires_grad=True
        )

        # self.positives = torch.randn(1, int(self.n_facts), 16)
        # self.positives /=  torch.sqrt(torch.sum(self.positives**2, dim=-1, keepdim=True))
        # self.negatives = - self.positives
        # self.positives = self.positives.to(device=self.device)
        # self.negatives = self.negatives.to(device=self.device)

        self.negative_scale = 1 * torch.ones(1, device=self.device)
        self.shift = torch.ones(0, device=self.device)

    def batchwise_cdist(self, samples1, samples2, eps=1e-6):
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

        return {"CS": pCs, "YS": pred, "pCS": pCs, "MUS": mus, "LOGVARS": logvars}

    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z"""
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

    def get_loss(self, args):
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            return ADDMNIST_SAT_AGG(ADDMNIST_Cumulative, self.task, pcbm=True)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        self.opt = torch.optim.Adam(
            [*self.parameters(), self.positives, self.negatives], args.lr
        )


def sample_gaussian_tensors(mu, logsigma, num_samples):
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
