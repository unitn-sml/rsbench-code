# Module which identifies an LTN loss
import ltn
import torch


class SDDOIA_SAT_AGG(torch.nn.Module):
    def __init__(self, And, Or, Not, Implies, Equiv, Forall) -> None:
        super().__init__()

        self.And = And
        self.Or = Or
        self.Not = Not
        self.Implies = Implies
        self.Equiv = Equiv
        self.Forall = Forall
        self.SatAgg = ltn.fuzzy_ops.SatAgg()

    def forward(self, out_dict, args, b_idx=None):
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
        conc_preds = out_dict["CS"]

        sat_loss = SDDOIAsat_agg_loss(self, conc_preds, Ys, b_idx)

        return sat_loss


def SDDOIAsat_agg_loss(self, conc_preds, actions, b_idx):
    """SDDOIA sat agg loss

    Args:
        eltn: eltn
        pCs: probability of the concepts
        labels: labels
        grade: grade

    Returns:
        loss: loss value
    """

    # Variables
    green_light = ltn.Variable("green_light", conc_preds[:, 0], add_batch_dim=False)
    follow = ltn.Variable("follow", conc_preds[:, 1], add_batch_dim=False)
    road_clear = ltn.Variable("road_clear", conc_preds[:, 2], add_batch_dim=False)
    red_light = ltn.Variable("red_light", conc_preds[:, 3], add_batch_dim=False)
    stop_sign = ltn.Variable("stop_sign", conc_preds[:, 4], add_batch_dim=False)
    car = ltn.Variable("car", conc_preds[:, 5], add_batch_dim=False)
    person = ltn.Variable("person", conc_preds[:, 6], add_batch_dim=False)
    rider = ltn.Variable("rider", conc_preds[:, 7], add_batch_dim=False)
    other_obstacle = ltn.Variable(
        "other_obstacle", conc_preds[:, 8], add_batch_dim=False
    )
    left_lane = ltn.Variable("left_lane", conc_preds[:, 9], add_batch_dim=False)
    left_green_light = ltn.Variable(
        "left_green_light", conc_preds[:, 10], add_batch_dim=False
    )
    left_follow = ltn.Variable("left_follow", conc_preds[:, 11], add_batch_dim=False)
    no_left_lane = ltn.Variable("no_left_lane", conc_preds[:, 12], add_batch_dim=False)
    left_obstacle = ltn.Variable(
        "left_obstacle", conc_preds[:, 13], add_batch_dim=False
    )
    left_solid_line = ltn.Variable(
        "left_solid_line", conc_preds[:, 14], add_batch_dim=False
    )
    right_lane = ltn.Variable("right_lane", conc_preds[:, 15], add_batch_dim=False)
    right_green_light = ltn.Variable(
        "right_green_light", conc_preds[:, 16], add_batch_dim=False
    )
    right_follow = ltn.Variable("right_follow", conc_preds[:, 17], add_batch_dim=False)
    no_right_lane = ltn.Variable(
        "no_right_lane", conc_preds[:, 18], add_batch_dim=False
    )
    right_obstacle = ltn.Variable(
        "right_obstacle", conc_preds[:, 19], add_batch_dim=False
    )
    right_solid_line = ltn.Variable(
        "right_solid_line", conc_preds[:, 20], add_batch_dim=False
    )
    move_forward = ltn.Variable("move_forward", actions[:, 0], add_batch_dim=False)
    stop = ltn.Variable("stop", actions[:, 1], add_batch_dim=False)
    turn_left = ltn.Variable("turn_left", actions[:, 2], add_batch_dim=False)
    turn_right = ltn.Variable("turn_right", actions[:, 3], add_batch_dim=False)

    # REDLIGHT: red_light ⇒ ¬green_light
    # phi1 = self.Forall(ltn.diag(red_light, green_light), self.Implies(red_light, self.Not(green_light)))
    phi1 = self.Forall(
        ltn.diag(red_light, green_light), self.Not(self.And(red_light, green_light))
    )
    # OBSTACLE: obstacle = car ∨ person ∨ rider ∨ other_obstacle

    def obstacle(c, p, r, o):
        return self.Or(c, self.Or(p, self.Or(r, o)))

    # ROAD_CLEAR: road_clear ⇐⇒ ¬obstacle
    phi2 = self.Forall(
        ltn.diag(road_clear, car, person, rider, other_obstacle),
        self.Equiv(road_clear, self.Not(obstacle(car, person, rider, other_obstacle))),
    )
    # MOVE_FORWARD: green_light ∨ follow ∨ clear ⇒ move_forward
    phi3 = self.Forall(
        ltn.diag(green_light, follow, road_clear, move_forward),
        self.Implies(self.Or(green_light, self.Or(follow, road_clear)), move_forward),
    )
    # phi3 = self.Forall(ltn.diag(green_light, follow, road_clear, move_forward),
    #                    self.Equiv(
    #                        self.Or(
    #                            green_light,
    #                            self.Or(
    #                                follow,
    #                                road_clear
    #                            )
    #                        ),
    #                        move_forward))
    # STOP: red_light ∨ stop_sign ∨ obstacle ⇒ stop
    phi4 = self.Forall(
        ltn.diag(red_light, stop_sign, car, person, rider, other_obstacle, stop),
        self.Implies(
            self.Or(
                red_light,
                self.Or(stop_sign, obstacle(car, person, rider, other_obstacle)),
            ),
            stop,
        ),
    )
    # phi4 = self.Forall(ltn.diag(red_light, stop_sign, car, person, rider, other_obstacle, stop),
    #                    self.Equiv(
    #                        self.Or(
    #                            red_light,
    #                            self.Or(
    #                                stop_sign,
    #                                obstacle(car, person, rider, other_obstacle)
    #                            )
    #                        ),
    #                        stop
    #                    ))
    # MOVE_FORWARD_2: stop ⇒ ¬move_forward
    # phi5 = self.Forall(ltn.diag(stop, move_forward),
    #                    self.Implies(
    #                        stop,
    #                        self.Not(move_forward)
    #                    ))
    phi5 = self.Forall(
        ltn.diag(
            red_light,
            stop_sign,
            car,
            person,
            rider,
            other_obstacle,
            green_light,
            follow,
            road_clear,
        ),
        self.Not(
            self.And(
                self.Or(
                    red_light,
                    self.Or(stop_sign, obstacle(car, person, rider, other_obstacle)),
                ),
                self.Or(green_light, self.Or(follow, road_clear)),
            )
        ),
    )
    # phi5 = self.Forall(ltn.diag(stop, move_forward),
    #                    self.Not(
    #                        self.And(
    #                            stop,
    #                            move_forward
    #                        )
    #                    ))

    # LEFT CAN TURN: can_turn = left_lane ∨ left_green_lane ∨ left_follow
    def can_turn(lane, green_light, follow):
        return self.Or(lane, self.Or(green_light, follow))

    # LEFT CANNOT TURN: cannot_turn = no_left_lane ∨ left_obstacle ∨ left_solid_line
    def cannot_turn(no_lane, obstacle, solid_line):
        return self.Or(no_lane, self.Or(obstacle, solid_line))

    # TURN LEFT: can_turn ∧ ¬cannot_turn ⇒ turn_left
    # phi6 = self.Forall(ltn.diag(left_lane, left_green_light, left_follow, no_left_lane, left_obstacle,
    #                             left_solid_line, turn_left),
    #                    self.Implies(
    #                        self.And(
    #                             can_turn(left_lane, left_green_light, left_follow),
    #                             self.Not(cannot_turn(no_left_lane, left_obstacle, left_solid_line))
    #                        ),
    #                        turn_left
    #                    ))
    phi6 = self.Forall(
        ltn.diag(left_lane, left_green_light, left_follow, turn_left),
        self.Equiv(can_turn(left_lane, left_green_light, left_follow), turn_left),
    )
    phi7 = self.Forall(
        ltn.diag(no_left_lane, left_obstacle, left_solid_line, turn_left),
        self.Equiv(
            self.Not(cannot_turn(no_left_lane, left_obstacle, left_solid_line)),
            turn_left,
        ),
    )
    # phi6 = self.Forall(ltn.diag(left_lane, left_green_light, left_follow, no_left_lane, left_obstacle,
    #                             left_solid_line, turn_left),
    #                    self.Equiv(
    #                        self.And(
    #                            can_turn(left_lane, left_green_light, left_follow),
    #                            self.Not(cannot_turn(no_left_lane, left_obstacle, left_solid_line))
    #                        ),
    #                        turn_left
    #                    ))

    # TURN RIGHT: can_turn ∧ ¬cannot_turn ⇒ turn_right
    # phi7 = self.Forall(ltn.diag(right_lane, right_green_light, right_follow, no_right_lane, right_obstacle,
    #                             right_solid_line, turn_right),
    #                    self.Implies(
    #                        self.And(
    #                            can_turn(right_lane, right_green_light, right_follow),
    #                            self.Not(cannot_turn(no_right_lane, right_obstacle, right_solid_line))
    #                        ),
    #                        turn_right
    #                    ))
    phi8 = self.Forall(
        ltn.diag(right_lane, right_green_light, right_follow, turn_right),
        self.Equiv(can_turn(right_lane, right_green_light, right_follow), turn_right),
    )
    phi9 = self.Forall(
        ltn.diag(no_right_lane, right_obstacle, right_solid_line, turn_right),
        self.Equiv(
            self.Not(cannot_turn(no_right_lane, right_obstacle, right_solid_line)),
            turn_right,
        ),
    )
    # phi7 = self.Forall(ltn.diag(right_lane, right_green_light, right_follow, no_right_lane, right_obstacle,
    #                             right_solid_line, turn_right),
    #                    self.Equiv(
    #                        self.And(
    #                            can_turn(right_lane, right_green_light, right_follow),
    #                            self.Not(cannot_turn(no_right_lane, right_obstacle, right_solid_line))
    #                        ),
    #                        turn_right
    #                    ))

    if b_idx == 0:
        print("phi1: " + str(phi1))
        print("phi2: " + str(phi2))
        print("phi3: " + str(phi3))
        print("phi4: " + str(phi4))
        print("phi5: " + str(phi5))
        print("phi6: " + str(phi6))
        print("phi7: " + str(phi7))
        print("phi8: " + str(phi8))
        print("phi9: " + str(phi9))
        print("\n")

    return 1.0 - self.SatAgg(phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9), None
