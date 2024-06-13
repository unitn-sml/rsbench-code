# Module which identifies an LTN loss
import ltn
import torch


class KAND_SAT_AGG(torch.nn.Module):
    def __init__(self, And, Or, Not, Forall, Equiv) -> None:
        super().__init__()
        self.nr_classes = 2
        self.And = And
        self.Or = Or
        self.Not = Not
        self.Forall = Forall
        self.Equiv = Equiv
        self.SatAgg = ltn.fuzzy_ops.SatAgg()

    def forward(self, out_dict, args, b_idx=None):
        # load from dict
        Ys = out_dict["LABELS"]  # groundtruth labels
        pCs = out_dict["pCS"]
        shape, color = torch.split(pCs, 3, dim=-1)

        if args.c_sup_ltn:
            sat_loss = self.KANDsat_agg_loss(
                shape, color, out_dict["conc_preds"], Ys[:, -1], args, b_idx
            )
        else:
            sat_loss = self.KANDsat_agg_loss(shape, color, None, Ys[:, -1], args, b_idx)

        return sat_loss

    def KANDsat_agg_loss(self, shapes, colors, conc_preds, labels, args, b_idx):
        args.print = False
        f1_s = ltn.Variable("f1_s", shapes[:, 0])
        f2_s = ltn.Variable("f2_s", shapes[:, 1])
        f3_s = ltn.Variable("f3_s", shapes[:, 2])
        f1_c = ltn.Variable("f1_c", colors[:, 0])
        f2_c = ltn.Variable("f2_c", colors[:, 1])
        f3_c = ltn.Variable("f3_c", colors[:, 2])
        l0 = ltn.Constant(torch.tensor(0))
        l1 = ltn.Constant(torch.tensor(1))
        l2 = ltn.Constant(torch.tensor(2))
        res = ltn.Variable("res", labels, add_batch_dim=False)

        get_obj = ltn.Function(func=lambda fig, o_idx: fig[:, o_idx[0]])
        get_conc = ltn.Predicate(func=lambda obj, c_idx: obj[:, c_idx[0]])

        if args.c_sup_ltn:
            square = ltn.Variable("square", conc_preds[0][:, :3])
            not_square = ltn.Variable("not_square", conc_preds[1][:, :3])
            red = ltn.Variable("red", conc_preds[2][:, 3:])
            not_red = ltn.Variable("not_red", conc_preds[3][:, 3:])
            phi1 = self.Forall(square, get_conc(square, l0), p=2)
            phi2 = self.Forall(red, get_conc(red, l0), p=2)
            phi3 = self.Forall(not_square, self.Not(get_conc(not_square, l0)), p=2)
            phi4 = self.Forall(not_red, self.Not(get_conc(not_red, l0)), p=2)

        # SAME concept

        and_0 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l0), get_conc(get_obj(f, l1), l0)),
            get_conc(get_obj(f, l2), l0),
        )

        and_1 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l1), get_conc(get_obj(f, l1), l1)),
            get_conc(get_obj(f, l2), l1),
        )

        and_2 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l2), get_conc(get_obj(f, l1), l2)),
            get_conc(get_obj(f, l2), l2),
        )

        def same(f, concept_type):
            value = self.Or(self.Or(and_0(f), and_1(f)), and_2(f))
            if args.print:
                print("same for %s: " % (concept_type,) + str(value))
            return value

        # DIFF concept

        _and_0 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l0), get_conc(get_obj(f, l1), l1)),
            get_conc(get_obj(f, l2), l2),
        )

        _and_1 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l0), get_conc(get_obj(f, l1), l2)),
            get_conc(get_obj(f, l2), l1),
        )

        _and_2 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l1), get_conc(get_obj(f, l1), l0)),
            get_conc(get_obj(f, l2), l2),
        )

        _and_3 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l1), get_conc(get_obj(f, l1), l2)),
            get_conc(get_obj(f, l2), l0),
        )

        _and_4 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l2), get_conc(get_obj(f, l1), l1)),
            get_conc(get_obj(f, l2), l0),
        )

        _and_5 = lambda f: self.And(
            self.And(get_conc(get_obj(f, l0), l2), get_conc(get_obj(f, l1), l0)),
            get_conc(get_obj(f, l2), l1),
        )

        def diff(f, concept_type):
            value = self.Or(
                _and_0(f),
                self.Or(
                    _and_1(f),
                    self.Or(
                        _and_2(f), self.Or(_and_3(f), self.Or(_and_4(f), _and_5(f)))
                    ),
                ),
            )
            if args.print:
                print("diff for %s: " % (concept_type,) + str(value))
            return value

        # PAIR concept

        def pair(f, concept_type):
            value = self.And(
                self.Not(same(f, concept_type)), self.Not(diff(f, concept_type))
            )
            if args.print:
                print("pair for %s: " % (concept_type,) + str(value))
            return value

        # final formulas

        # inner formula

        def and_same(f1, f2, f3, concept_type):
            value = self.And(
                same(f1, concept_type),
                self.And(same(f2, concept_type), same(f3, concept_type)),
            )
            if args.print:
                print("and_same for %s: " % (concept_type,) + str(value))
            return value

        def and_pair(f1, f2, f3, concept_type):
            value = self.And(
                pair(f1, concept_type),
                self.And(pair(f2, concept_type), pair(f3, concept_type)),
            )
            if args.print:
                print("and_pair for %s: " % (concept_type,) + str(value))
            return value

        def and_diff(f1, f2, f3, concept_type):
            value = self.And(
                diff(f1, concept_type),
                self.And(diff(f2, concept_type), diff(f3, concept_type)),
            )
            if args.print:
                print("and_diff for %s: " % (concept_type,) + str(value))
            return value

        def or_formula(f1, f2, f3, concept_type):
            value = self.Or(
                and_same(f1, f2, f3, concept_type),
                self.Or(
                    and_pair(f1, f2, f3, concept_type),
                    and_diff(f1, f2, f3, concept_type),
                ),
            )
            if args.print:
                print("or_formula for %s: " % (concept_type,) + str(value))
            return value

        def final_formula(f1_s, f2_s, f3_s, f1_c, f2_c, f3_c):
            value = self.Or(
                or_formula(f1_s, f2_s, f3_s, "shape"),
                or_formula(f1_c, f2_c, f3_c, "color"),
            )
            if args.print:
                print("final_formula: " + str(value))
            return value

        phi5 = self.Forall(
            ltn.diag(f1_s, f2_s, f3_s, f1_c, f2_c, f3_c, res),
            self.Equiv(final_formula(f1_s, f2_s, f3_s, f1_c, f2_c, f3_c), res),
        )

        if args.c_sup_ltn:
            sat_agg = self.SatAgg(phi1, phi2, phi3, phi4, phi5)
        else:
            sat_agg = phi5.value

        if b_idx == 0 and args.c_sup_ltn:
            print("Square: " + str(phi1))
            print("Not square: " + str(phi2))
            print("Red: " + str(phi3))
            print("Not red: " + str(phi4))
            print("Formula: " + str(phi5))
            print("\n")

        if args.print:
            print("sat agg value: " + str(sat_agg))

        return 1 - sat_agg, None
