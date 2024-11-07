# LTN architecture for BOIA
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from utils.boia_ltn_loss import SDDOIA_SAT_AGG
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


class BOIALTN(CExt):
    """LTN architecture for BOIA"""

    NAME = "boialtn"
    """
    BOIA in LTN
    """

    def __init__(self, encoder, n_images=2, c_split=(), args=None):
        super(BOIALTN, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )
        # device
        self.device = get_device()
        self.n_facts = 21
        self.nr_classes = 4

        # how many images and explicit split of concepts
        self.c_split = c_split

        # opt and device
        self.opt = None
        self.boia_ood_knowledge = args.boia_ood_knowledge

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

        # compute the cs
        cs = torch.squeeze(cs, dim=-1)

        # compute predictions
        cs_preds = cs >= 0.5
        green_light = cs_preds[:, 0]
        follow = cs_preds[:, 1]
        road_clear = cs_preds[:, 2]
        red_light = cs_preds[:, 3]
        stop_sign = cs_preds[:, 4]
        car = cs_preds[:, 5]
        person = cs_preds[:, 6]
        rider = cs_preds[:, 7]
        other_obstacle = cs_preds[:, 8]
        left_lane = cs_preds[:, 9]
        left_green_light = cs_preds[:, 10]
        left_follow = cs_preds[:, 11]
        right_lane = cs_preds[:, 15]
        right_green_light = cs_preds[:, 16]
        right_follow = cs_preds[:, 17]
        no_left_lane = cs_preds[:, 12]
        left_obstacle = cs_preds[:, 13]
        left_solid_line = cs_preds[:, 14]
        no_right_lane = cs_preds[:, 18]
        right_obstacle = cs_preds[:, 19]
        right_solid_line = cs_preds[:, 20]

        if not self.boia_ood_knowledge:
            move_forward = torch.logical_or(
                torch.logical_or(green_light, follow), road_clear
            )
            stop = torch.logical_or(
                red_light,
                torch.logical_or(
                    stop_sign,
                    torch.logical_or(
                        car,
                        torch.logical_or(
                            person, torch.logical_or(rider, other_obstacle)
                        ),
                    ),
                ),
            )
            left_can_turn = torch.logical_or(
                left_lane, torch.logical_or(left_green_light, left_follow)
            )
            right_can_turn = torch.logical_or(
                right_lane, torch.logical_or(right_green_light, right_follow)
            )
            left_cannot_turn = torch.logical_or(
                no_left_lane, torch.logical_or(left_obstacle, left_solid_line)
            )
            right_cannot_turn = torch.logical_or(
                no_right_lane, torch.logical_or(right_obstacle, right_solid_line)
            )
            turn_left = torch.logical_and(
                left_can_turn, torch.logical_not(left_cannot_turn)
            )
            turn_right = torch.logical_and(
                right_can_turn, torch.logical_not(right_cannot_turn)
            )

        else:
            stop = torch.logical_or(
                car,
                torch.logical_or(person, torch.logical_or(rider, other_obstacle)),
            )
            move_forward = torch.logical_not(stop)

            left_can_turn = left_lane
            right_can_turn = right_lane

            left_cannot_turn = torch.logical_or(no_left_lane, left_obstacle)
            right_cannot_turn = torch.logical_or(no_right_lane, right_obstacle)

            turn_left = torch.logical_and(
                left_can_turn, torch.logical_not(left_cannot_turn)
            )
            turn_right = torch.logical_and(
                right_can_turn, torch.logical_not(right_cannot_turn)
            )

        preds = torch.stack((move_forward, stop, turn_left, turn_right), dim=1)
        return {
            "CS": cs,
            "YS": torch.nn.functional.one_hot(preds.long(), num_classes=2)
            .reshape(-1, 8)
            .long(),
            "pCS": pCs,
        }

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
            _and, _implies = None, None
            if args.and_op == "Godel":
                _and = ltn.fuzzy_ops.AndMin()
            elif args.and_op == "Prod":
                _and = ltn.fuzzy_ops.AndProd()
            else:
                _and = ltn.fuzzy_ops.AndLuk()
            if args.or_op == "Godel":
                Or = ltn.Connective(ltn.fuzzy_ops.OrMax())
            elif args.or_op == "Prod":
                Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
            else:
                Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
            if args.imp_op == "Godel":
                _implies = ltn.fuzzy_ops.ImpliesGodel()
            elif args.imp_op == "Prod":
                _implies = ltn.fuzzy_ops.ImpliesReichenbach()
            elif args.imp_op == "Luk":
                _implies = ltn.fuzzy_ops.ImpliesLuk()
            elif args.imp_op == "Goguen":
                _implies = ltn.fuzzy_ops.ImpliesGoguen()
            else:
                _implies = ltn.fuzzy_ops.ImpliesKleeneDienes()

            And = ltn.Connective(_and)
            Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
            Implies = ltn.Connective(_implies)
            Equiv = ltn.Connective(
                ltn.fuzzy_ops.Equiv(and_op=_and, implies_op=_implies)
            )
            Forall = ltn.Quantifier(
                ltn.fuzzy_ops.AggregPMeanError(p=args.p), quantifier="f"
            )
            return SDDOIA_SAT_AGG(And, Or, Not, Implies, Equiv, Forall)
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
