# DPL model for MINIBOIA
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import MINIBOIA_Cumulative
from utils.dpl_loss import MINIBOIA_DPL


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MiniBoiaDPL(DeepProblogModel):
    """DPL MODEL FOR MINI BOIA"""

    NAME = "miniboiadpl"

    """
    BOIA but with synthetic data
    """

    def __init__(
        self,
        encoder,
        n_images=1,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=20,
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
        super(MiniBoiaDPL, self).__init__(
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

        # logic
        logic = create_w_to_y()
        self.or_four_bits = logic.to(self.device)

        # Worlds-queries matrix
        if self.args.task == "boia":

            if args.boia_ood_knowledge:
                # build the world query matrices for mini boia
                self.FS_w_q = build_world_queries_matrix_FS_ambulance().to(self.device)
                # self.LR_w_q = build_world_queries_matrix_LR().to(self.device)
                self.L_w_q = build_world_queries_matrix_L_ambulance().to(self.device)
                self.R_w_q = build_world_queries_matrix_R_ambulance().to(self.device)
            else:
                # build the world query matrices for mini boia
                self.FS_w_q = build_world_queries_matrix_FS().to(self.device)
                # self.LR_w_q = build_world_queries_matrix_LR().to(self.device)
                self.L_w_q = build_world_queries_matrix_L().to(self.device)
                self.R_w_q = build_world_queries_matrix_R().to(self.device)
        else:
            raise NotImplementedError("Invalid task for MINIBOIA")

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

        # Problog inference to compute worlds and query probability distributions
        py = self.problog_inference(pCs)

        cs = torch.squeeze(cs, dim=-1)

        return {"CS": cs, "YS": py, "pCS": pCs}

    def problog_inference(self, pCs, query=None):
        """Performs ProbLog inference to retrieve the worlds probability distribution P(w). Works with two encoded bits.

        Args:
            self: instance
            pCs: probability of concepts
            query (default=None): query

        Returns:
            query_prob: query probability
            worlds_prob: worlds probability
        """

        # for forward
        tl_green = pCs[:, :2]  # traffic light is green
        follow = pCs[:, 2:4]  # follow car ahead
        clear = pCs[:, 4:6]  # road is clear

        # for stop
        tl_red = pCs[:, 6:8]  # traffic light is red
        t_sign = pCs[:, 8:10]  # traffic sign present
        obs = compute_logic_obstacle(self.or_four_bits, pCs)  # generic obstacle

        A = tl_green.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        B = follow.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        C = clear.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        D = tl_red.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
        E = t_sign.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        F = obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_FS = (
            A.multiply(B).multiply(C).multiply(D).multiply(E).multiply(F).view(-1, 64)
        )
        #
        labels_FS = torch.einsum("bi,ik->bk", w_FS, self.FS_w_q)
        ##

        # for LEFT
        left_lane = pCs[:, 18:20]  # there is LEFT lane
        tl_green_left = pCs[:, 20:22]  # tl green on LEFT
        follow_left = pCs[:, 22:24]  # follow car going LEFT

        # for LEFT-STOP
        no_left_lane = pCs[:, 24:26]  # no lane on LEFT
        l_obs = pCs[:, 26:28]  # LEFT obstacle
        left_line = pCs[:, 28:30]  # solid line on LEFT

        AL = left_lane.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        BL = (
            tl_green_left.unsqueeze(1)
            .unsqueeze(3)
            .unsqueeze(4)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        CL = (
            follow_left.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        )
        DL = (
            no_left_lane.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        EL = l_obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        FL = left_line.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_L = (
            AL.multiply(BL)
            .multiply(CL)
            .multiply(DL)
            .multiply(EL)
            .multiply(FL)
            .view(-1, 64)
        )

        label_L = torch.einsum("bi,ik->bk", w_L, self.L_w_q)
        ##

        # for RIGHT
        rigt_lane = pCs[:, 30:32]  # there is RIGHT lane
        tl_green_rigt = pCs[:, 32:34]  # tl green on RIGHT
        follow_rigt = pCs[:, 34:36]  # follow car going RIGHT

        # for RIGHT-STOP
        no_rigt_lane = pCs[:, 36:38]  # no lane on RIGHT
        r_obs = pCs[:, 38:40]  # RIGHT obstacle
        rigt_line = pCs[:, 40:42]  # solid line on RIGHT

        AL = rigt_lane.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        BL = (
            tl_green_rigt.unsqueeze(1)
            .unsqueeze(3)
            .unsqueeze(4)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        CL = (
            follow_rigt.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        )
        DL = (
            no_rigt_lane.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        EL = r_obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        FL = rigt_line.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_R = (
            AL.multiply(BL)
            .multiply(CL)
            .multiply(DL)
            .multiply(EL)
            .multiply(FL)
            .view(-1, 64)
        )

        label_R = torch.einsum("bi,ik->bk", w_R, self.R_w_q)

        pred = torch.cat([labels_FS, label_L, label_R], dim=1)  # this is 8 dim

        # avoid overflow
        pred = (pred + 1e-5) / (1 + 2 * 1e-5)

        return pred

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
        if args.dataset in ["miniboia", "preminiboia"]:
            return MINIBOIA_DPL(MINIBOIA_Cumulative)
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
        self.or_four_bits = self.or_four_bits.to(device)

        # Worlds-queries matrix
        if self.args.task == "boia":
            self.FS_w_q = self.FS_w_q.to(device)
            self.L_w_q = self.L_w_q.to(device)
            self.R_w_q = self.R_w_q.to(device)
