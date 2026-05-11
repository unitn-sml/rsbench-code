# RAVEN for DPL
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.deepproblog_modules import DeepProblogModel
from models.utils.utils_problog import build_worlds_queries_matrix_RAVEN
from models.utils.ops import outer_product
from utils.args import *
from utils.conf import get_device
from utils.dpl_loss import RAVEN_DPL
from utils.losses import RAVEN_Cumulative


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class RavenDPL(DeepProblogModel):
    """RAVEN DeepProbLog model with factorized logic per attribute."""

    NAME = "ravendpl"

    def __init__(
        self,
        encoder,
        n_images=16,
        c_split=(),
        model_dict=None,
        n_facts=9,
        nr_classes=8,
        args=None,
    ):
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=16): number of images (8 context + 8 choices)
            c_split: concept splits
            model_dict (default=None): model dictionary
            n_facts (int, default=9): number of concepts (3+3+3 for RAVEN-3x3x3)
            nr_classes (int, default=8): number of choice candidates
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        super(RavenDPL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # Concept dimensions per attribute
        self.raven_config = args.raven_config if args else "center_single"
        if self.raven_config == "center_single":
            self.dims = {"Type": 3, "Size": 3, "Color": 3}
            self.n_facts = 9  # 3+3+3

        self.nr_classes = nr_classes  # 8 candidate choices
        self.n_and_classes = 2  # invalid / valid for explicit row-rule agreement

        # Per-attribute rules.
        # In RAVEN-3x3x3 we also support Distribute_Three. For a 3-value domain,
        # this means the row contains all three distinct values in some order.
        # Arithmetic does not apply to Type.
        self.attr_rules = {}
        for attr in self.dims:
            if attr == "Type":
                self.attr_rules[attr] = ["Constant", "Progression", "Distribute_Three"]
            else:
                self.attr_rules[attr] = [
                    "Constant",
                    "Progression",
                    "Arithmetic",
                    "Distribute_Three",
                ]

        # Worlds-queries matrices (one per attribute, with attribute-specific rules)
        # plus attribute-specific explicit row-agreement tables over rule pairs.
        self.wq_matrices = {}
        self.and_rules = {}
        self.device = get_device()
        for attr, dim in self.dims.items():
            rules = self.attr_rules[attr]
            mat, and_rule = build_worlds_queries_matrix_RAVEN(dim, rules)
            self.wq_matrices[attr] = mat.float().to(self.device)
            self.and_rules[attr] = and_rule.float().to(self.device)
            

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector [batch, 16, 1, 160, 160]

        Returns:
            out_dict: output dictionary
        """
        # 1. Encoding
        z, _ = self.encoder(x)  # [B, 16, n_facts]

        # 2. Concept Extraction (softmax + clamp per attribute)
        concepts_probs = self.normalize_concepts(z)

        # Concatenate all attribute probs for supervision
        extracted_probs = [concepts_probs[attr] for attr in self.dims]
        pCS = torch.cat(extracted_probs, dim=-1)  # [B, 16, n_facts]

        # 3. Factorized Symbolic Reasoning
        # Compute row-level rule distributions for the two observed context rows.
        row_rule_dists = {}
        for attr, probs in concepts_probs.items():
            wq = self.wq_matrices[attr]

            # Row 1: panels [0, 1, 2] -> rule distribution [B, n_rules]
            p_rule_r1 = self.problog_inference(probs[:, [0, 1, 2], :], wq)
            # Row 2: panels [3, 4, 5] -> rule distribution [B, n_rules]
            p_rule_r2 = self.problog_inference(probs[:, [3, 4, 5], :], wq)

            row_rule_dists[attr] = {"r1": p_rule_r1, "r2": p_rule_r2}

        # Score the 8 candidate choices via explicit 3-row agreement.
        ys = self.score_choices(concepts_probs, row_rule_dists)

        # Prepare output dictionary
        out_dict = {"YS": ys, "CS": z, "pCS": pCS}

        # Add row-level rule distributions for debugging/analysis.
        for attr, row_dists in row_rule_dists.items():
            out_dict[f"{attr}_R1_PREDS"] = row_dists["r1"]
            out_dict[f"{attr}_R2_PREDS"] = row_dists["r2"]

        return out_dict

    def normalize_concepts(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z,
        applying softmax with epsilon clamping per attribute group.

        Args:
            self: instance
            z (torch.tensor): encoder output [B, 16, n_facts]

        Returns:
            concepts_probs: dict mapping attribute name -> [B, 16, dim] probability tensor
        """

        def soft_clamp(h, dim=-1):
            h = nn.Softmax(dim=dim)(h)
            eps = 1e-5
            h = h + eps
            with torch.no_grad():
                Z = torch.sum(h, dim=dim, keepdim=True)
            h = h / Z
            return h

        # Split z into per-attribute logits and apply softmax + clamp
        concepts_probs = {}
        offset = 0
        for attr, dim in self.dims.items():
            concepts_probs[attr] = soft_clamp(z[..., offset : offset + dim])
            offset += dim

        return concepts_probs

    def problog_inference(self, row_probs, wq):
        """Problog inference for a single attribute over a single row of 3 panels.

        Computes the outer product of 3 panel concept distributions to build
        the world probability vector, then maps to rule distributions via w_q.

        Args:
            self: instance
            row_probs (torch.tensor): [B, 3, n_vals] concept probs for 3 panels
            wq (torch.tensor): [n_vals^3, n_rules] logic matrix

        Returns:
            rule_dist: [B, n_rules] probability of each rule given this row
        """
        # Number of rules is determined by the w_q matrix (varies per attribute)
        n_rules = wq.shape[1]

        # Outer product of 3 panel distributions -> world probabilities
        worlds_tensor = outer_product(
            row_probs[:, 0], row_probs[:, 1], row_probs[:, 2]
        )  # [B, N, N, N]

        worlds_prob = worlds_tensor.reshape(worlds_tensor.shape[0], -1)  # [B, N^3]

        # Compute rule distribution: sum over worlds weighted by w_q
        rule_dist = torch.zeros(
            size=(worlds_prob.shape[0], n_rules), device=worlds_prob.device
        )
        for i in range(n_rules):
            rule_dist[:, i] = self.compute_query(i, worlds_prob, wq).view(-1)

        return rule_dist

    def problog_inference_log(self, row_log_probs, wq):
        """Log-space ProbLog inference for a single attribute over a single row.

        Computes log P(rule) = logsumexp over worlds where the rule holds,
        using log-space throughout to avoid gradient starvation from
        near-uniform probability distributions.

        Args:
            self: instance
            row_log_probs (torch.tensor): [B, 3, n_vals] log concept probs for 3 panels
            wq (torch.tensor): [n_vals^3, n_rules] binary logic matrix

        Returns:
            log_rule_dist: [B, n_rules] log-probability of each rule given this row
        """
        n_rules = wq.shape[1]
        n_worlds = wq.shape[0]
        batch_size = row_log_probs.shape[0]

        # Sum log-probs across 3 panels for each world (AND in log-space)
        # row_log_probs: [B, 3, n_vals]
        # worlds_log_prob[b, i*j*k] = row_log_probs[b,0,i] + row_log_probs[b,1,j] + row_log_probs[b,2,k]
        log_p0 = row_log_probs[:, 0, :]  # [B, n_vals]
        log_p1 = row_log_probs[:, 1, :]  # [B, n_vals]
        log_p2 = row_log_probs[:, 2, :]  # [B, n_vals]

        # Compute all world log-probs via broadcast
        # [B, n_vals, 1, 1] + [B, 1, n_vals, 1] + [B, 1, 1, n_vals] -> [B, n_vals, n_vals, n_vals]
        worlds_log = log_p0.unsqueeze(2).unsqueeze(3) + log_p1.unsqueeze(1).unsqueeze(3) + log_p2.unsqueeze(1).unsqueeze(2)
        worlds_log = worlds_log.reshape(batch_size, n_worlds)  # [B, n_vals^3]

        # Compute log rule distribution: logsumexp over worlds where rule holds
        log_rule_dist = torch.full(
            (batch_size, n_rules), -1e10, device=worlds_log.device
        )
        for i in range(n_rules):
            # Mask: which worlds have this rule
            mask = wq[:, i].bool()  # [n_worlds]
            if mask.any():
                # logsumexp over the masked worlds
                masked_log = worlds_log[:, mask]  # [B, n_valid_worlds]
                log_rule_dist[:, i] = torch.logsumexp(masked_log, dim=1)

        return log_rule_dist

    def compute_query(self, query, worlds_prob, wq):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query (int): query index (rule index)
            worlds_prob (torch.tensor): [B, N^3] world probabilities
            wq (torch.tensor): [N^3, n_rules] logic matrix

        Returns:
            query_prob: [B, 1] probability of the query
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = wq[:, query]
        # Compute query probability by summing the probability of all worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def score_choices(self, concepts_probs, row_rule_dists):
        """Score the 8 candidate choices in log-space via explicit 3-row rule agreement.

        For each candidate and attribute, the model computes the log rule
        distributions of row 1, row 2, and the candidate-completed row 3,
        builds a joint world over ``(r1, r2, r3)``, and uses an attribute-specific
        ``and_rule`` to compute the log-probability of agreement (r1==r2==r3).

        Log-space scoring avoids gradient starvation: when concept probabilities
        are near-uniform, probability-space scores for all candidates are nearly
        identical (differing by ~3e-5), producing vanishing gradients. In log
        space, the same differences are amplified ~9x per attribute, keeping
        the gradient signal alive through the reasoning layer.

        This mirrors how KandDPL (the rsbench Kand-Logic model) computes its
        output: raw probabilities from the reasoning layer, with log applied in
        the loss function. No intermediate softmax is applied.

        Args:
            self: instance
            concepts_probs: dict mapping attribute name -> [B, 16, dim] probs
            row_rule_dists: dict mapping attribute name ->
                {"r1": [B, n_rules], "r2": [B, n_rules]}  (probability-space, kept for analysis)

        Returns:
            ys: [B, 8] log-softmaxed choice log-probabilities
        """
        # Compute log concept probabilities
        concepts_log_probs = {
            attr: probs.clamp(min=1e-8).log()
            for attr, probs in concepts_probs.items()
        }

        # Pre-compute log rule distributions for context rows (in log-space)
        row_log_rule_dists = {}
        for attr, log_probs in concepts_log_probs.items():
            wq = self.wq_matrices[attr]
            log_rule_r1 = self.problog_inference_log(
                log_probs[:, [0, 1, 2], :], wq
            )
            log_rule_r2 = self.problog_inference_log(
                log_probs[:, [3, 4, 5], :], wq
            )
            row_log_rule_dists[attr] = {"r1": log_rule_r1, "r2": log_rule_r2}

        # Score each candidate
        choice_log_scores = []

        for c_idx in range(8, 16):
            attr_log_scores = []

            for attr, log_probs in concepts_log_probs.items():
                wq = self.wq_matrices[attr]
                log_rule_r1 = row_log_rule_dists[attr]["r1"]
                log_rule_r2 = row_log_rule_dists[attr]["r2"]

                # Candidate-completed Row 3 in log-space
                log_rule_r3 = self.problog_inference_log(
                    log_probs[:, [6, 7, c_idx], :], wq
                )  # [B, n_rules]

                # Compute log agreement probability
                log_agreement = self.compute_three_row_rule_agreement_log(
                    log_rule_r1, log_rule_r2, log_rule_r3, attr
                )  # [B]

                attr_log_scores.append(log_agreement)

            # Combine attribute log-scores by summing (AND across independent attributes)
            total_log_score = torch.stack(attr_log_scores, dim=1).sum(dim=1)  # [B]
            choice_log_scores.append(total_log_score)

        log_scores = torch.stack(choice_log_scores, dim=1)  # [B, 8]

        # Return log-softmax (no intermediate exp/softmax that kills gradients)
        ys = F.log_softmax(log_scores, dim=-1)

        return ys

    def compute_three_row_rule_agreement_log(self, log_rule_r1, log_rule_r2, log_rule_r3, attr):
        """Compute log-probability of agreement (r1==r2==r3) in log-space.

        For each valid rule triplet (r, r, r), computes:
            log P(r1=r, r2=r, r3=r) = log_rule_r1[r] + log_rule_r2[r] + log_rule_r3[r]
        Then marginalizes over all valid triplets via logsumexp.

        Args:
            self: instance
            log_rule_r1 (torch.tensor): [B, n_rules] log rule distribution for row 1
            log_rule_r2 (torch.tensor): [B, n_rules] log rule distribution for row 2
            log_rule_r3 (torch.tensor): [B, n_rules] log rule distribution for row 3
            attr (str): attribute name

        Returns:
            log_agreement (torch.tensor): [B] log-probability of valid agreement
        """
        and_rule = self.and_rules[attr]  # [n_rules^3, 2]
        # Valid triplets: where and_rule[:, 1] == 1
        valid_mask = and_rule[:, 1].bool()  # [n_rules^3]

        if not valid_mask.any():
            return torch.full(
                (log_rule_r1.shape[0],), -1e10, device=log_rule_r1.device
            )

        n_rules = log_rule_r1.shape[1]

        # Build log-probability for each rule triplet world
        # log P(r1, r2, r3) = log_rule_r1[r1] + log_rule_r2[r2] + log_rule_r3[r3]
        # Using broadcast: [B, n_rules, 1, 1] + [B, 1, n_rules, 1] + [B, 1, 1, n_rules]
        log_triplet = (
            log_rule_r1.unsqueeze(2).unsqueeze(3)
            + log_rule_r2.unsqueeze(1).unsqueeze(3)
            + log_rule_r3.unsqueeze(1).unsqueeze(2)
        )  # [B, n_rules, n_rules, n_rules]
        log_triplet = log_triplet.reshape(log_rule_r1.shape[0], n_rules**3)  # [B, n_rules^3]

        # Select only valid triplets and logsumexp
        log_valid_triplets = log_triplet[:, valid_mask]  # [B, n_valid]
        log_agreement = torch.logsumexp(log_valid_triplets, dim=1)  # [B]

        return log_agreement

    def compute_three_row_rule_agreement(self, p_rule_r1, p_rule_r2, p_rule_r3, attr):
        """Compute explicit agreement over the latent rules of rows 1, 2, and 3.

        Kept for backward compatibility and analysis (confusion matrices, etc.).
        The log-space version ``compute_three_row_rule_agreement_log`` is used
        for training.

        Args:
            self: instance
            p_rule_r1 (torch.tensor): [B, n_rules] rule distribution for row 1
            p_rule_r2 (torch.tensor): [B, n_rules] rule distribution for row 2
            p_rule_r3 (torch.tensor): [B, n_rules] rule distribution for the
                candidate-completed row 3
            attr (str): attribute name (`Type`, `Size`, or `Color`)

        Returns:
            agreement_probs (torch.tensor): [B, 2] where column 0 is invalid
                (`r1, r2, r3` not all equal) and column 1 is valid
                (`r1 == r2 == r3`)
        """
        n_rules = p_rule_r1.shape[1]
        assert p_rule_r2.shape[1] == n_rules and p_rule_r3.shape[1] == n_rules, (
            p_rule_r1.shape,
            p_rule_r2.shape,
            p_rule_r3.shape,
        )

        rule_triplet_worlds = outer_product(
            p_rule_r1, p_rule_r2, p_rule_r3
        ).reshape(-1, n_rules**3)

        agreement_probs = torch.zeros(
            size=(p_rule_r1.shape[0], self.n_and_classes), device=p_rule_r1.device
        )

        and_rule = self.and_rules[attr]
        for i in range(self.n_and_classes):
            agreement_probs[:, i] = torch.sum(
                and_rule[:, i] * rule_triplet_worlds, dim=1
            )

        return agreement_probs



    @staticmethod
    def get_loss(args):
        """Returns the loss function

        Args:
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if dataset is not specified
        """
        if args.dataset == "raven":
            return RAVEN_DPL(RAVEN_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Starts the optimizer

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(),
            args.lr,
            weight_decay=1e-5,
        )

    # override of to
    def to(self, device):
        super().to(device)
        for attr in self.dims:
            self.wq_matrices[attr] = self.wq_matrices[attr].to(device)
            self.and_rules[attr] = self.and_rules[attr].to(device)
        return self
