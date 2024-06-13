# Module which identifies an LTN loss
import ltn
import torch
import itertools
from utils.normal_kl_divergence import kl_divergence


class ADDMNIST_SAT_AGG(torch.nn.Module):
    def __init__(self, And, Exists, Forall, task="addition") -> None:
        super().__init__()
        self.task = task
        self.And = And
        self.Exists = Exists
        self.Forall = Forall

        if task == "addition":
            self.nr_classes = 19
        elif task == "product":
            self.nr_classes = 81
        elif task == "multiop":
            self.nr_classes = 15

    def forward(self, out_dict, args):
        """Forward module

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
        """
        # load from dict
        Ys = out_dict["LABELS"]
        pCs = out_dict["pCS"]

        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1, :]

        if self.task == "addition":
            sat_loss = self.ADDMNISTsat_agg_loss(prob_digit1, prob_digit2, Ys)
        elif self.task == "product":
            sat_loss = PRODMNISTsat_agg_loss(prob_digit1, prob_digit2, Ys)
        elif self.task == "multiop":
            sat_loss = MULTIOPsat_agg_loss(prob_digit1, prob_digit2, Ys)

        return sat_loss

    def ADDMNISTsat_agg_loss(self, p1, p2, labels):
        """Addmnist sat agg loss

        Args:
            p1: probability of the first concept
            p2: probability of the second concept
            labels: ground truth labels
        Returns:
            loss: loss value
        """
        # convert to variables
        x = ltn.Variable("x", p1)  # , trainable=True)
        y = ltn.Variable("y", p2)  # , trainable=True)
        n = ltn.Variable("n", labels)

        # LTN predicate for getting correct digit
        Digit = ltn.Predicate(func=lambda digits, d_idx: torch.gather(digits, 1, d_idx))

        # variables in LTN
        d1 = ltn.Variable("d1", torch.tensor(range(p1.shape[-1])))
        d2 = ltn.Variable("d2", torch.tensor(range(p1.shape[-1])))

        sat_agg = self.Forall(
            ltn.diag(x, y, n),
            self.Exists(
                [d1, d2],
                self.And(Digit(x, d1), Digit(y, d2)),
                cond_vars=[d1, d2, n],
                cond_fn=lambda d1, d2, n: torch.eq((d1.value + d2.value), n.value),
            ),
        )

        return 1 - sat_agg.value, None


def PRODMNISTsat_agg_loss(eltn, p1, p2, labels, grade):
    """Prodmnist sat agg loss

    Args:
        eltn: eltn
        p1: probability of the first concept
        p2: probability of the second concept
        labels: labels
        grade: grade

    Returns:
        loss: loss value
    """
    max_c = p1.size(-1)

    # convert to variables
    bit1 = ltn.Variable("bit1", p1)  # , trainable=True)
    bit2 = ltn.Variable("bit2", p2)  # , trainable=True)
    y_true = ltn.Variable("labels", labels)

    # print(bit1)
    # print(bit2)
    # print(y_true)

    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))

    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
        ltn.diag(bit1, bit2, y_true),
        Exists(
            [b_1, b_2],
            And(eltn(bit1, b_1), eltn(bit2, b_2)),
            cond_vars=[b_1, b_2, y_true],
            cond_fn=lambda b_1, b_2, z: torch.eq(b_1.value * b_2.value, z.value),
            p=grade,
        ),
    ).value

    return 1 - sat_agg


def MULTIOPsat_agg_loss(eltn, p1, p2, labels, grade):
    """Multioperation sat agg loss

    Args:
        eltn: eltn
        p1: probability of the first concept
        p2: probability of the second concept
        labels: labels
        grade: grade

    Returns:
        loss: loss value
    """
    max_c = p1.size(-1)

    # convert to variables
    bit1 = ltn.Variable("bit1", p1)  # , trainable=True)
    bit2 = ltn.Variable("bit2", p2)  # , trainable=True)
    y_true = ltn.Variable("labels", labels)

    # variables in LTN
    b_1 = ltn.Variable("b_1", torch.tensor(range(max_c)))
    b_2 = ltn.Variable("b_2", torch.tensor(range(max_c)))

    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    sat_agg = Forall(
        ltn.diag(bit1, bit2, y_true),
        Exists(
            [b_1, b_2],
            And(eltn(bit1, b_1), eltn(bit2, b_2)),
            cond_vars=[b_1, b_2, y_true],
            cond_fn=lambda b_1, b_2, z: torch.eq(
                b_1.value**2 + b_2.value**2 + b_1.value * b_2.value, z.value
            ),
            p=grade,
        ),
    ).value
    return 1 - sat_agg
