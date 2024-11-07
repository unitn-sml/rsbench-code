import os.path
import random
from datetime import datetime
from itertools import product
from math import isnan
from pathlib import Path
from time import time, sleep
import numpy as np
import pandas as pd
import torch
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD
from torch import nn
import itertools

from problog.logic import Term, Constant
from problog.logic import Var, AnnotatedDisjunction


def lock_resource(lock_filename):
    with open(lock_filename, "w") as f:
        f.write("locked")


def release_lock(lock_filename):
    os.remove(lock_filename)


def create_facts(sequence_len, n_digits=10):
    """
    Return the list of ADs necessary to describe an image with 'sequence_len' digits.
    'n_facts' specifies how many digits we are considering (i.e. n_facts = 2 means that the images can contain only 0 or 1)
    """

    ad = []  # Empty list to store the ADs
    for i in range(sequence_len):
        pos = i + 1
        annot_disj = ""  # Empty string to store the current AD facts

        # Build the AD
        digit = Term("digit")
        X = Var("X")
        facts = [
            digit(X, Constant(pos), Constant(y), p="p_" + str(pos) + str(y))
            for y in range(n_digits)
        ]
        annot_disj += str(AnnotatedDisjunction(facts, None)) + "."

        ad.append(annot_disj)

    return ad


def define_ProbLog_model(facts, rules, label, digit_query=None, mode="query"):
    """Build the ProbLog model using teh given facts, rules, evidence and query."""
    model = ""  # Empty program

    # Insert annotated disjuctions
    for i in range(len(facts)):
        model += "\n\n% Digit in position " + str(i + 1) + "\n\n"
        model += facts[i]

    # Insert rules
    model += "\n\n% Rules\n"
    model += rules

    # Insert digit query
    if digit_query:
        model += "\n\n% Digit Query\n"
        model += "query(" + digit_query + ")."

    # Insert addition query
    if mode == "query":
        model += "\n\n% Addition Query\n"
        model += "query(addition(img," + str(label) + "))."

    elif mode == "evidence":
        model += "\n\n% Addition Evidence\n"
        model += "evidence(addition(img," + str(label) + "))."

    return model


def update_resource(log_filepath, update_info, lock_filename="access.lock"):
    # {'Experiment_ID': 0, 'Run_ID': 1, ...}
    print("Updating resource with: {}".format(update_info))

    # Check if lock file does exist
    # If it exists -> I have to wait (sleep -> 1.0 second)
    while os.path.isfile(lock_filename):
        sleep(1.0)

    # Do lock
    lock_resource(lock_filename)

    # Do update
    try:
        log_file = open(log_filepath, "a")
        log_file.write(update_info)
        log_file.close()
    except Exception as e:
        raise e
    finally:
        # Release lock
        release_lock(lock_filename)


def load_mnist_classifier(checkpoint_path, device):
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    clf = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1),
    )

    if torch.cuda.is_available():
        clf.load_state_dict(torch.load(checkpoint_path))
        clf = clf.to(device)
    else:
        clf.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device("cpu"))
        )

    return clf


def define_experiment(exp_folder, exp_class, params, exp_counter):
    log_file = Path(os.path.join(exp_folder, exp_class, exp_class + ".csv"))
    params_columns = [
        "latent_dim_sub",
        "latent_dim_sym",
        "learning_rate",
        "dropout",
        "dropout_ENC",
        "dropout_DEC",
        "recon_w",
        "kl_w",
        "query_w",
        "sup_w",
    ]
    if log_file.is_file():
        # Load file
        log_csv = pd.read_csv(os.path.join(exp_folder, exp_class, exp_class + ".csv"))

        # Check if the required number of test has been already satisfied
        required_exp = params["n_exp"]

        if len(log_csv) > 0:
            query = "".join(f" {key} == {params[key]} &" for key in params_columns)[:-1]
            n_exp = len(log_csv.query(query))
            if n_exp == 0:
                exp_ID = log_csv["exp_ID"].max() + 1
                if isnan(exp_ID):
                    exp_ID = 1
                counter = required_exp - n_exp
                print(
                    "\n\n{} compatible experiments found in file {} -> {} experiments to run.".format(
                        n_exp,
                        os.path.join(exp_folder, exp_class, exp_class + ".csv"),
                        counter,
                    )
                )

                run_ID = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")
            elif n_exp < required_exp:
                exp_ID = log_csv.query(query)["exp_ID"].values[0]
                counter = required_exp - n_exp
                print(
                    "\n\n{} compatible experiments found in file {} -> {} experiments to run.".format(
                        n_exp,
                        os.path.join(exp_folder, exp_class, exp_class + ".csv"),
                        counter,
                    )
                )

                run_ID = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")

            else:
                print(
                    "\n\n{} compatible experiments found in file {} -> No experiments to run.".format(
                        n_exp,
                        os.path.join(exp_folder, exp_class, exp_class + ".csv"),
                        0,
                    )
                )
                counter = 0
                exp_ID = log_csv.query(query)["exp_ID"].values[0]
                run_ID = None
        else:
            counter = required_exp
            exp_ID = log_csv["exp_ID"].max() + 1
            if isnan(exp_ID):
                exp_ID = 1
            run_ID = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")
            print(
                "\n\n0 compatible experiments found in file {} -> {} experiments to run.".format(
                    exp_folder + exp_class + ".csv", counter
                )
            )

    else:
        counter = params["n_exp"]
        # Create log file
        log_file = open(os.path.join(exp_folder, exp_class, exp_class + ".csv"), "w")
        header = (
            "exp_ID,run_ID,"
            + "".join(str(key) + "," for key in params_columns)
            + params["rec_loss"]
            + "_recon_val,acc_discr_val,"
            + params["rec_loss"]
            + "_recon_test,acc_discr_test,acc_gen,epochs,max_epoch,time,tag\n"
        )
        log_file.write(header)
        # Define experiment ID
        exp_ID = 1
        run_ID = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")
        print()
        print("-" * 40)
        print(
            "\nNO csv file found -> new file created {}".format(
                os.path.join(exp_folder, exp_class, exp_class + ".csv")
            )
        )
        print("-" * 40)
        print()
        log_file.close()

    exp_counter += 1
    print()
    print("*" * 40)
    print("Running exp {} (exp ID: {})".format(exp_counter, exp_ID))
    print("Parameters:", params)
    print("*" * 40)
    print()

    return run_ID, str(exp_ID), exp_counter, counter, params_columns


def build_model_dict(sequence_len, n_digits):
    """Define dictionary of pre-compiled ProbLog models"""
    possible_query_add = {2: list(range(0, (n_digits - 1) * 2 + 1))}
    rules = "addition(X,N) :- digit(X,1,N1), digit(X,2,N2), N is N1 + N2.\ndigits(X,Y):-digit(img,1,X), digit(img,2,Y)."
    facts = create_facts(sequence_len, n_digits=n_digits)
    model_dict = {
        "query": {add: "EMPTY" for add in possible_query_add[sequence_len]},
        "evidence": {add: "EMPTY" for add in possible_query_add[sequence_len]},
    }

    for mode in ["query", "evidence"]:
        for add in model_dict[mode]:
            problog_model = define_ProbLog_model(
                facts, rules, label=add, digit_query="digits(X,Y)", mode=mode
            )
            lf = LogicFormula.create_from(problog_model)
            dag = LogicDAG.create_from(lf)
            sdd = SDD.create_from(dag)
            model_dict[mode][add] = sdd

    return model_dict


def build_worlds_queries_matrix_kandinsky(sequence_len=0, n_facts=0, n_shapes=0):
    """Build Worlds Queries Matrices

    The Kandinsky Figure has two pairs of objects with the same shape,
        in one pair the objects have the same color,
        in the other pair different colors
    """
    from collections import Counter, defaultdict

    def find_equal_indices(vector):
        index_dict = defaultdict(list)
        for i, value in enumerate(vector):
            index_dict[value].append(i)

        equal_indices = {
            key: indices for key, indices in index_dict.items() if len(indices) > 1
        }
        return list(equal_indices.values())

    def two_pairs(shapesf1, shapesf2):
        c_f1 = Counter(shapesf1)
        c_f2 = Counter(shapesf2)

        idx_1, idx_2 = find_equal_indices(shapesf1), find_equal_indices(shapesf2)
        to_rtn_idx = [idx_1, idx_2]

        if len(c_f1) == 2 and len(c_f2) == 2:
            for _, value in c_f1.items():
                if value != 2:
                    return False, to_rtn_idx
            for _, value in c_f2.items():
                if value != 2:
                    return False, to_rtn_idx
            return True, to_rtn_idx
        return False, to_rtn_idx

    possible_worlds = list(
        product(range(n_facts * n_shapes), repeat=sequence_len)
    )  # 576
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, 2))  # 2 possible queries (it is or it is not, right?)
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)  # (576, 2)

    for w in range(n_worlds):
        figure1, figure2 = look_up[w]
        """
        shape1f1, shape2f1, shape3f1, shape4f1 = get_shape(figure1)
        shape2f2, shape2f2, shape3f2, shape4f2 = get_shape(figure2)
        color1f1, color2f1, color3f1, color4f1 = get_color(figure1)
        color2f2, color2f2, color3f2, color4f2 = get_color(figure2)

        # kandinsky condition
        cond, pairsf1, pairsf2 = two_pairs([shape1f1, shape2f1, shape3f1, shape4f1], [shape2f2, shape2f2, shape3f2, shape4f2])
        """

        # if digit1 + digit2 == q:
        #    w_q[w, q] = 1
    return w_q


def build_worlds_queries_matrix(sequence_len=0, n_digits=0, task="addmnist"):
    """Build Worlds-Queries matrix"""
    if task == "addmnist":
        possible_worlds = list(product(range(n_digits), repeat=sequence_len))
        n_worlds = len(possible_worlds)
        n_queries = len(range(0, 10 + 10))
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
        for w in range(n_worlds):
            digit1, digit2 = look_up[w]
            for q in range(n_queries):
                if digit1 + digit2 == q:
                    w_q[w, q] = 1
        return w_q

    elif task == "productmnist":
        possible_worlds = list(product(range(n_digits), repeat=sequence_len))
        n_worlds = len(possible_worlds)
        n_queries = [0]
        for i, j in itertools.product(range(1, 10), range(1, 10)):
            n_queries.append(i * j)
        n_queries = np.unique(np.array(n_queries))

        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, len(n_queries))  # (100, boh)
        for w in range(n_worlds):
            digit1, digit2 = look_up[w]
            for i, q in enumerate(n_queries):
                if digit1 * digit2 == q:
                    w_q[w, i] = 1

        return w_q

    elif task == "multiopmnist":
        possible_worlds = list(product(range(n_digits), repeat=sequence_len))
        n_worlds = len(possible_worlds)
        n_queries = np.array([0, 1, 2, 3])

        w_q = torch.zeros(n_worlds, len(n_queries))  # (16, 4)
        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        for w in range(n_worlds):
            digit1, digit2 = look_up[w]
            for i, q in enumerate(n_queries):
                if digit1 + digit2 == 1 and digit1 * digit2 == 0:
                    w_q[w, 0] = 1
                elif digit1 + digit2 == 2 and digit1 * digit2 == 0:
                    w_q[w, 1] = 1
                elif digit1 + digit2 == 4 and digit1 * digit2 == 3:
                    w_q[w, 2] = 1
                else:
                    w_q[w, 3] = 1
        return w_q

    else:
        NotImplementedError("Wrong choice")


def build_worlds_queries_matrix_KAND(
    n_images=3, n_concepts=6, n_poss=3, task="mini_patterns"
):
    """Build Worlds-Queries matrix"""

    and_rule = torch.zeros((2**n_images, 2))
    and_rule[:-1] = torch.tensor([1, 0])
    and_rule[-1] = torch.tensor([0, 1])

    if task == "mini_patterns":

        or_rule = torch.zeros((2**2, 2))
        or_rule[0, 0] = 1
        or_rule[1:, 1] = 1

        and_rule = torch.zeros((3**n_images, 2))

        possible_preds = list(product(range(3), repeat=3))
        n_preds = len(possible_preds)

        preds = {i: c for i, c in zip(range(n_preds), possible_preds)}
        for p in range(n_preds):
            res1, res2, res3 = preds[p]

            if (res1 == res2) and (res2 == res3):
                and_rule[p, 1] = 1
            else:
                and_rule[p, 0] = 1

        possible_worlds = list(product(range(3), repeat=3))
        n_worlds = len(possible_worlds)

        n_queries = 3

        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, n_queries)  # (3^6, 9)
        for w in range(n_worlds):
            s1, s2, s3 = look_up[w]

            if (s1 == s2) and (s1 == s3):
                w_q[w, 2] = 1
            elif (s1 != s2) and (s1 != s3) and (s2 != s3):
                w_q[w, 0] = 1
            else:
                w_q[w, 1] = 1

        return w_q, and_rule, or_rule

    elif task == "patterns":

        and_or_rule = torch.zeros((9**n_images, 2))

        possible_preds = list(product(range(9), repeat=3))
        n_preds = len(possible_preds)

        preds = {i: c for i, c in zip(range(n_preds), possible_preds)}
        for p in range(n_preds):
            im1, im2, im3 = preds[p]
            if (im1 == im2) and (im1 == im3) and (im2 == im3):
                and_or_rule[p, 1] = 1
            else:
                and_or_rule[p, 0] = 1

        possible_worlds = list(product(range(n_poss), repeat=n_concepts))
        n_worlds = len(possible_worlds)

        n_queries = 9

        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, n_queries)  # (3^6, 9)
        for w in range(n_worlds):
            s1, s2, s3, c1, c2, c3 = look_up[w]

            same_s = (s1 == s2) and (s1 == s3)
            diff_s = (s1 != s2) and (s1 != s3) and (s2 != s3)
            pair_s = not same_s and not diff_s

            shapes = np.array([diff_s, pair_s, same_s])

            same_c = (c1 == c2) and (c1 == c3)
            diff_c = (c1 != c2) and (c1 != c3) and (c2 != c3)
            pair_c = not same_c and not diff_c

            colors = np.array([diff_c, pair_c, same_c])

            y = 3 * np.argmax(shapes) + np.argmax(colors)

            w_q[w, y] = 1

        return w_q, and_or_rule

    elif task == "red_triangle":
        possible_worlds = list(product(range(n_poss), repeat=n_concepts))
        n_worlds = len(possible_worlds)

        n_queries = 2

        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, n_queries)  # (3^8, 2)
        for w in range(n_worlds):
            s1, s2, s3, c1, c2, c3 = look_up[w]

            rt1 = (s1 == 0) and (c1 == 0)
            rt2 = (s2 == 0) and (c2 == 0)
            rt3 = (s3 == 0) and (c3 == 0)

            if rt1 or rt2 or rt3:
                w_q[w, 1] = 1
            else:
                w_q[w, 0] = 1

        return w_q, and_rule

    elif task == "base":
        possible_worlds = list(product(range(n_poss), repeat=n_concepts))
        n_worlds = len(possible_worlds)

        n_queries = 2

        look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
        w_q = torch.zeros(n_worlds, n_queries)  # (3^8, 2)
        for w in range(n_worlds):
            s1, s2, s3, s4, c1, c2, c3, c4 = look_up[w]

            s12 = s1 == s2
            s13 = s1 == s3
            s14 = s1 == s4
            c12 = c1 == c2
            c13 = c1 == c3
            c14 = c1 == c4

            s23 = s2 == s3
            s24 = s2 == s4
            c23 = c2 == c3
            c24 = c2 == c4

            s34 = s3 == s4
            c34 = c3 == c4

            p0 = s12 * s34 * ((c12 + c34) % 2) * (s1 != s3)
            p1 = s13 * s24 * ((c13 + c24) % 2) * (s1 != s2)
            p2 = s14 * s23 * ((c14 + c23) % 2) * (s1 != s2)

            if p0 + p1 + p2 == 0:
                w_q[w, 0] = 1
            else:
                w_q[w, 1] = 1

        return w_q, and_rule

    else:
        NotImplementedError("Wrong choice")


def build_clevr_worlds_queries_matrix(max_n_images=4, n_concepts=19):
    pass


if __name__ == "__main__":

    w_q, and_or_rule = build_worlds_queries_matrix_KAND(task="mini_patterns")

    # print(w_q)

    # possible_worlds = list(product(range(3), repeat=6))

    # # print(possible_worlds)
    # count = np.zeros((9,))
    # for i in range(len(w_q)):
    #     # if w_q[i, 1] == 1:
    #     print('(',  possible_worlds[i][0], possible_worlds[i][0+3], ')',
    #           '(',  possible_worlds[i][1], possible_worlds[i][1+3], ')',
    #           '(',  possible_worlds[i][2], possible_worlds[i][2+3], ')',
    #         #   '(',  possible_worlds[i][3], possible_worlds[i][3+3], ')',
    #           '->', torch.argmax(w_q[i]) )
    #         # count += 1
    #     for j in range(w_q.shape[-1]):
    #         if w_q[i, j] == 1: count[j] += 1

    possible_worlds = list(product(range(9), repeat=3))

    # print(possible_worlds)
    count = np.zeros((9,))
    for i in range(len(w_q)):
        # if w_q[i, 1] == 1:
        print(
            possible_worlds[i],
            "  ",
            "(",
            possible_worlds[i][0] % 3,
            possible_worlds[i][0] // 3,
            ")",
            "(",
            possible_worlds[i][1] % 3,
            possible_worlds[i][1] // 3,
            ")",
            "(",
            possible_worlds[i][2] % 3,
            possible_worlds[i][2] // 3,
            ")",
            #   '(',  possible_worlds[i][3], possible_worlds[i][3+3], ')',
            "->",
            torch.argmax(and_or_rule[i]),
        )
        # count += 1
    # print('\n', count / 9**3) #), 'vs', len(w_q) - count )
    # print(np.sum(count) / 9**3)
    # print(9**4)

    # print( np.log10(count)*count + np.log10(len(w_q) - count)*(len(w_q) - count) )
    # print(9**4*np.log10(9**4))


def build_world_queries_matrix_complete_FS():

    possible_worlds = list(product(range(2), repeat=9))
    n_worlds = len(possible_worlds)
    n_queries = 4
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_green, follow, clear, tl_red, t_sign, ob1, ob2, ob3, ob4 = look_up[w]

        obs = min(ob1 + ob2 + ob3 + ob4, 1)

        if tl_green + follow + clear > 0:
            if tl_green + tl_red == 2 or clear + obs == 2:
                pass
            elif tl_red + t_sign + obs > 0:
                w_q[w, 0] = 1  # not move
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 1] = 1  # move forward
                w_q[w, 2] = 1  # no-stop
        else:
            w_q[w, 0] = 1  # not move
            if tl_red + t_sign + obs > 0:
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 2] = 1  # no-stop
    return w_q


def build_world_queries_matrix_FS():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 4
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_green, follow, clear, tl_red, t_sign, obs = look_up[w]

        if tl_green + follow + clear > 0:
            if tl_green + tl_red == 2 or clear + obs == 2:
                pass
            elif tl_red + t_sign + obs > 0:
                w_q[w, 0] = 1  # not move
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 1] = 1  # move forward
                w_q[w, 2] = 1  # no-stop
        else:
            w_q[w, 0] = 1  # not move
            if tl_red + t_sign + obs > 0:
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 2] = 1  # no-stop
    return w_q


# Case of the ambulance, predict forward
def build_world_queries_matrix_FS_ambulance():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 4
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_green, follow, clear, tl_red, t_sign, obs = look_up[w]

        # stop only if there is an obstacle
        if obs > 0:
            w_q[w, 0] = 1  # not move
            w_q[w, 3] = 1  # stop
        else:
            w_q[w, 1] = 1  # move forward
            w_q[w, 2] = 1  # no-stop
    return w_q


def build_world_queries_matrix_LR():

    possible_worlds = list(product(range(2), repeat=7))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_red, no_left_lane, left_solid_line, obs, left_lane, tl_green, follow = (
            look_up[w]
        )

        if left_lane + tl_green + follow > 0:
            if tl_green + tl_red == 2 or no_left_lane == 1:
                pass  # invalid
            elif tl_red + obs + left_solid_line > 0:
                w_q[w, 0] = 1  # not move
            else:
                w_q[w, 1] = 1  # move forward
        else:
            w_q[w, 0] = 1  # not move
    # for i in range(n_worlds):
    #     print(look_up[i], '->',w_q[i])
    return w_q


def build_world_queries_matrix_L():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        left_lane, tl_green, follow, no_left_lane, obs, left_solid_line = look_up[w]

        if left_lane + tl_green + follow + no_left_lane + obs + left_solid_line == 0:
            w_q[w, 0] = 0.5
            w_q[w, 1] = 0.5
        elif left_lane + tl_green + follow > 0:
            w_q[w, 1] = 1
        else:
            w_q[w, 0] = 1
    return w_q


# OOD knowledge (Ambulance)
def build_world_queries_matrix_L_ambulance():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        left_lane, tl_green, follow, no_left_lane, obs, left_solid_line = look_up[w]

        if no_left_lane + obs == 0:
            w_q[w, 0] = 1  # not turn
            w_q[w, 1] = 0  # turn
        elif left_lane > 0:
            w_q[w, 0] = 0  # not turn
            w_q[w, 1] = 1  # turn
        else:
            w_q[w, 0] = 1  # not turn
            w_q[w, 1] = 0  # turn
    return w_q


def build_world_queries_matrix_R():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        right_lane, tl_green, follow, no_right_lane, obs, right_solid_line = look_up[w]

        if right_lane + tl_green + follow + no_right_lane + obs + right_solid_line == 0:
            w_q[w, 0] = 0.5
            w_q[w, 1] = 0.5
        elif right_lane + tl_green + follow > 0:
            if obs + right_solid_line + no_right_lane > 0:
                w_q[w, 0] = 1  # not move
            else:
                w_q[w, 1] = 1  # move
        else:
            w_q[w, 0] = 1  # not move
    return w_q


def build_world_queries_matrix_R_ambulance():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        right_lane, tl_green, follow, no_right_lane, obs, right_solid_line = look_up[w]

        if no_right_lane + obs == 0:
            w_q[w, 0] = 1  # not turn
            w_q[w, 1] = 0  # turn
        elif right_lane > 0:
            w_q[w, 0] = 0  # not turn
            w_q[w, 1] = 1  # turn
        else:
            w_q[w, 0] = 1  # not turn
            w_q[w, 1] = 0  # turn
    return w_q


def compute_logic_forward(or_three_bits, concepts: torch.Tensor):
    A = concepts[:, :2].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 2:4].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 4:6].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 2 * 2 * 2)

    active = torch.einsum("bi,ik->bk", poss_worlds, or_three_bits)

    # assert torch.abs(active.sum() / len(active)- 1) < 0.001, (active, active.sum() / len(active) )

    return active


def compute_logic_stop(or_six_bits, concepts: torch.Tensor):
    A = (
        concepts[:, 6:8]
        .unsqueeze(2)
        .unsqueeze(3)
        .unsqueeze(4)
        .unsqueeze(5)
        .unsqueeze(6)
    )
    B = (
        concepts[:, 8:10]
        .unsqueeze(1)
        .unsqueeze(3)
        .unsqueeze(4)
        .unsqueeze(5)
        .unsqueeze(6)
    )
    C = (
        concepts[:, 10:12]
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(4)
        .unsqueeze(5)
        .unsqueeze(6)
    )
    D = (
        concepts[:, 12:14]
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .unsqueeze(5)
        .unsqueeze(6)
    )
    E = (
        concepts[:, 14:16]
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .unsqueeze(4)
        .unsqueeze(6)
    )
    F = (
        concepts[:, 16:18]
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .unsqueeze(4)
        .unsqueeze(5)
    )

    poss_worlds = (
        A.multiply(B).multiply(C).multiply(D).multiply(E).multiply(F).view(-1, 64)
    )

    active = torch.einsum("bi,ik->bk", poss_worlds, or_six_bits)

    return active


def compute_logic_no_left(or_three_bits, concepts: torch.Tensor):
    A = concepts[:, 18:20].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 20:22].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 22:24].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum("bi,ik->bk", poss_worlds, or_three_bits)

    return active


def compute_logic_left(or_three_bits, concepts: torch.Tensor):
    A = concepts[:, 24:26].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 26:28].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 28:30].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum("bi,ik->bk", poss_worlds, or_three_bits)

    return active


def compute_logic_no_right(or_three_bits, concepts: torch.Tensor):
    A = concepts[:, 30:32].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 32:34].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 34:36].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum("bi,ik->bk", poss_worlds, or_three_bits)

    return active


def compute_logic_right(or_three_bits, concepts: torch.Tensor):
    A = concepts[:, 36:38].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 38:40].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 40:42].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum("bi,ik->bk", poss_worlds, or_three_bits)

    return active


def compute_logic_no_left_lane(or_two_bits, pC):
    A = pC[:, 18:20].unsqueeze(2)  # no lane on left
    B = pC[:, 22:24].unsqueeze(1)  # solid line on left

    obs_worlds = A.multiply(B).view(-1, 4)

    no_left_lane = torch.einsum("bi,ik->bk", obs_worlds, or_two_bits)

    return no_left_lane


def compute_logic_obstacle(or_four_bits, pC):

    o_car = pC[:, 10:12].unsqueeze(2).unsqueeze(3).unsqueeze(4)  # car
    o_person = pC[:, 12:14].unsqueeze(1).unsqueeze(3).unsqueeze(4)  # person
    o_rider = pC[:, 14:16].unsqueeze(1).unsqueeze(2).unsqueeze(4)  # rider
    o_other = pC[:, 16:18].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # other obstacle

    obs_worlds = (
        o_car.multiply(o_person).multiply(o_rider).multiply(o_other).view(-1, 16)
    )

    obs_active = torch.einsum("bi,ik->bk", obs_worlds, or_four_bits)

    return obs_active


def create_w_to_y():

    four_bits_or = torch.cat((torch.zeros((16, 1)), torch.ones((16, 1))), dim=1).to(
        dtype=torch.float
    )
    four_bits_or[0] = torch.tensor([1, 0])

    return four_bits_or
