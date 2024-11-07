# This module contains the computation of the metrics

import torch
import numpy as np
import math
from numpy import ndarray
from typing import Tuple, Dict, List
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.special import softmax


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)

    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = pred.eq(target.long())

    # if topk = (1,5), then, k=1 and k=5
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_binary(y_pred, y_true):

    # Split predictions into separate tensors for each label
    y_pred_split = torch.split(y_pred, 2, dim=1)
    y_pred_binary = torch.stack([pred.argmax(dim=1) for pred in y_pred_split], dim=1)

    acc = (
        (y_true.detach().cpu() == y_pred_binary.detach().cpu())
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    f1 = f1_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy(), average="micro")

    return acc * 100, f1 * 100


def evaluate_mix(true, pred):
    """Evaluate f1 and accuracy

    Args:
        true: Groundtruth values
        pred: Predicted values

    Returns:
        ac: accuracy
        f1: f1 score
    """
    try:
        ac = accuracy_score(true, pred)
    except ValueError:
        ac = (true == pred).astype(float).mean().item()

    try:
        f1 = f1_score(true, pred, average="weighted")
    except ValueError:
        # Compute true positives, false positives, and false negatives
        tp = np.sum(pred & true, axis=1)
        fp = np.sum(pred & ~true, axis=1)
        fn = np.sum(~pred & true, axis=1)

        # Compute precision, recall, and F1 score for each sample
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Take the average F1 score across all samples
        f1 = np.mean(f1)

    return ac, f1


def evaluate_metrics(
    model,
    loader,
    args,
    last=False,
    concatenated_concepts=True,
    apply_softmax=False,
    cf1=False,
):
    """Evaluate metrics of the frequentist model

    Args:
        model (nn.Module): network
        loader: dataloader
        args: command line arguments
        last (bool, default=False): evaluate metrics on the test set
        concatenated_concepts (bool, default=True): return concatenated concepts
        apply_softmax (bool, default=False): whether to apply softmax

    Returns:
        y_true: groundtruth labels, if last is specified
        gs: groundtruth concepts, if last is specified
        ys: label predictions, if last is specified
        cs: concept predictions, if last is specified
        p_cs: concept probability, if last is specified
        p_ys: label probability, if last is specified
        p_cs_all: concept proabability all, if last is specified
        p_ys_all: label probability all, if last is specified
        tloss: test loss, otherwise
        cacc: concept accuracy, otherwise
        yacc: label accuracy, otherwise
        f1sc: f1 on concept, otherwise
    """
    L = len(loader)
    tloss, cacc, yacc = 0, 0, 0
    f1sc, f1 = 0, 0
    fcf1 = 0
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)
        out_dict.update({"INPUTS": images, "LABELS": labels, "CONCEPTS": concepts})

        if last and i == 0:
            y_true = labels.detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
            y_pred = out_dict["YS"].detach().cpu().numpy()
            c_pred = out_dict["CS"].detach().cpu().numpy()
            pc_pred = out_dict["pCS"].detach().cpu().numpy()
            if args.dataset in ["minikandinsky", "clipkandinksy"]:
                pc_pred = pc_pred.reshape(pc_pred.shape[0], pc_pred.shape[1], 18)
        elif last and i > 0:
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
            c_true = np.concatenate([c_true, concepts.detach().cpu().numpy()], axis=0)
            y_pred = np.concatenate(
                [y_pred, out_dict["YS"].detach().cpu().numpy()], axis=0
            )
            c_pred = np.concatenate(
                [c_pred, out_dict["CS"].detach().cpu().numpy()], axis=0
            )
            o = out_dict["pCS"].detach().cpu().numpy()
            if args.dataset in ["minikandinsky", "clipkandinsky"]:
                pc_pred = np.concatenate(
                    [pc_pred, o.reshape(o.shape[0], o.shape[1], 18)], axis=0
                )
            else:
                pc_pred = np.concatenate([pc_pred, o], axis=0)

        if (
            args.dataset
            in [
                "addmnist",
                "shortmnist",
                "restrictedmnist",
                "halfmnist",
                "clipshortmnist",
            ]
            and not last
        ):
            loss, ac, acc, f1 = ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts, args)
        elif args.dataset in [
            "kandinsky",
            "prekandinsky",
            "minikandinsky",
            "clipkandinsky",
        ]:
            loss, ac, acc, f1 = KAND_eval_tloss_cacc_acc(out_dict)
        elif args.dataset in [
            "sddoia",
            "boia",
            "presddoia",
            "clipboia",
            "clipsddoia",
        ]:
            loss, ac, acc, f1 = SDDOIA_eval_tloss_cacc_acc(out_dict)
        else:
            NotImplementedError()

        if not last:
            tloss += loss.item()
            cacc += ac
            yacc += acc
            f1sc += f1
            if cf1:
                fcf1 += rcf1

    if apply_softmax:
        y_pred = softmax(y_pred, axis=1)

    if last:

        if args.dataset in [
            "sddoia",
            "boia",
            "presddoia",
            "clipboia",
            "clipsddoia",
        ]:
            y_pred_split = np.split(y_pred, 4, axis=1)
            ys = np.stack([pred.argmax(axis=1) for pred in y_pred_split], axis=1)
            p_ys = np.stack([pred.max(axis=1) for pred in y_pred_split], axis=1)

            gs = c_true
            cs = c_pred
            p_cs = pc_pred
        else:
            ys = np.argmax(y_pred, axis=1)

            gs = np.split(c_true, c_true.shape[1], axis=1)
            cs = np.split(c_pred, c_pred.shape[1], axis=1)
            p_cs = np.split(pc_pred, pc_pred.shape[1], axis=1)
            p_ys = y_pred

        p_cs_all = p_cs
        p_ys_all = y_pred

        assert len(gs) == len(cs), f"gs: {gs.shape}, cs: {cs.shape}"

        if args.dataset not in [
            "sddoia",
            "boia",
            "presddoia",
            "clipboia",
            "clipsddoia",
        ]:
            gs = np.concatenate(gs, axis=0).squeeze(1)
        if args.dataset in [
            "sddoia",
            "boia",
            "presddoia",
            "clipboia",
            "clipsddoia",
        ]:
            cs = (cs >= 0.5).astype(np.int)
        elif args.dataset not in [
            "kandinsky",
            "prekandinsky",
            "minikandinsky",
            "clipkandinsky",
        ]:
            cs = np.concatenate(cs, axis=0).squeeze(1).argmax(axis=-1)
            p_cs = (
                np.concatenate(p_cs, axis=0).squeeze(1).max(axis=-1)
            )  # should take the maximum as it is the associated probability
        else:
            cs = np.concatenate(cs, axis=0).squeeze(1)

            p_cs = np.concatenate(p_cs, axis=0).squeeze(1).max(axis=-1)

            cs = np.split(cs, 6, axis=-1)
            p_cs = np.split(p_cs, 6, axis=-1)

            cs = np.concatenate(
                [np.argmax(c, axis=-1).reshape(-1, 1) for c in cs], axis=-1
            )
            p_cs = np.concatenate(
                [np.argmax(pc, axis=-1).reshape(-1, 1) for pc in p_cs], axis=-1
            )

        if args.dataset not in [
            "sddoia",
            "boia",
            "presddoia",
            "clipboia",
            "clipsddoia",
        ]:
            p_cs_all = np.concatenate(p_cs_all, axis=0).squeeze(
                1
            )  # all the items of probabilities are considered
            p_ys = p_ys.max(axis=1)

        assert gs.shape == cs.shape, f"gs: {gs.shape}, cs: {cs.shape}"

        if not concatenated_concepts:
            # by default, consider the concept to return as separated internally
            gs = c_true
            cs = c_pred.argmax(axis=2)

        return y_true, gs, ys, cs, p_cs, p_ys, p_cs_all, p_ys_all
    else:
        if args.dataset in [
            "kandinsky",
            "prekandinsky",
            "minikandinsky",
            "clipkandinsky",
            "sddoia",
            "boia",
            "clipboia",
            "presddoia",
            "addmnist",
            "halfmnist",
            "shortmnist",
            "restrictedmnist",
            "clipsddoia",
            "clipshortmnist",
        ]:
            if cf1:
                return tloss / L, cacc / L, yacc / L, f1sc / L, fcf1 / L
            else:
                return tloss / L, cacc / L, yacc / L, f1sc / L
        else:
            return tloss / L, cacc / L, yacc / L, 0


def ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts, args):
    """ADDMMNIST evaluation

    Args:
        out_dict (Dict[str]): dictionary of outputs
        concepts: concepts

    Returns:
        loss: loss
        cacc: concept accuracy
        acc: label accuracy
        zero: 0
    """
    reprs = out_dict["CS"]
    L = len(reprs)

    objs = torch.split(
        reprs,
        1,
        dim=1,
    )
    g_objs = torch.split(concepts, 1, dim=1)

    if args.dataset != "clipshortmnist" and args.dataset != "shortmnist":

        assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

        loss, cacc = 0, 0
        for j in range(len(objs)):
            # enconding + ground truth
            obj_enc = objs[j].squeeze(dim=1)
            g_obj = g_objs[j].to(torch.long).view(-1)

            # evaluate loss
            loss += torch.nn.CrossEntropyLoss()(obj_enc, g_obj)

            # concept accuracy of object
            c_pred = torch.argmax(obj_enc, dim=1)

            assert (
                c_pred.size() == g_obj.size()
            ), f"size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}"

            correct = (c_pred == g_obj).sum().item()
            cacc += correct / len(objs[j])
    else:
        cacc = 0.0
        loss = torch.tensor([0.0])

    y = out_dict["YS"]
    y_true = out_dict["LABELS"]

    y_pred = torch.argmax(y, dim=-1)
    assert (
        y_pred.size() == y_true.size()
    ), f"size c_pred: {c_pred.size()}, size g_objs: {g_obj.size()}"

    acc = (y_pred.detach().cpu() == y_true.detach().cpu()).sum().item() / len(y_true)

    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")

    return loss / len(objs), cacc / len(objs) * 100, acc * 100, f1 * 100


def KAND_eval_tloss_cacc_acc(out_dict, debug=True, cf1=False):
    """KAND evaluation

    Args:
        out_dict (Dict[str]): dictionary of outputs
        debug: debug mode

    Returns:
        loss: loss
        cacc: concept accuracy
        acc: label accuracy
        zero: 0
    """
    reprs = out_dict["CS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)

    n_objs = len(g_objs)

    loss = torch.tensor(0.0, device=reprs.device)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    cacc, n_concepts, c_pred, rcf1 = 0, 0, None, 0
    for j in range(n_objs):

        cs = torch.split(objs[j], 3, dim=-1)
        gs = torch.split(g_objs[j], 1, dim=-1)

        n_concepts = len(gs)

        assert len(cs) == len(gs), f"{len(cs)}-{len(gs)}"

        for k in range(n_concepts):
            # print(cs[k].squeeze(1).shape, gs[k].shape)
            target = gs[k].view(-1)

            # loss += torch.nn.CrossEntropyLoss()(cs[k].squeeze(1),
            #                                     target.view(-1))
            # concept accuracy of object
            c_pred = torch.argmax(cs[k].squeeze(1), dim=-1)

            assert (
                c_pred.size() == target.size()
            ), f"size c_pred: {c_pred.size()}, size target: {target.size()}"

            correct = (c_pred == target).sum().item()
            cacc += correct / len(target)

            rcf1 = f1_score(c_pred.cpu().numpy(), target.cpu().numpy(), average="macro")

    cacc /= n_objs * n_concepts
    rcf1 /= n_objs * n_concepts

    y = out_dict["YS"]
    y_true = out_dict["LABELS"][:, -1]

    y_pred = torch.argmax(y, dim=-1)
    assert (
        y_pred.size() == y_true.size()
    ), f"size c_pred: {c_pred.size()}, size g_objs: {g_objs.size()}"

    acc = (y_pred.detach().cpu() == y_true.detach().cpu()).sum().item() / len(y_true)

    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")

    if cf1:
        return loss / len(objs), cacc * 100, acc * 100, f1 * 100, rcf1 * 100
    else:
        return loss / len(objs), cacc * 100, acc * 100, f1 * 100


def SDDOIA_eval_tloss_cacc_acc(out_dict):
    """SDDOIA evaluation

    Args:
        out_dict (Dict[str]): dictionary of outputs

    Returns:
        loss: loss
        cacc: concept accuracy
        acc: label accuracy
        f1: f1 accuracy
    """
    cs = out_dict["CS"]
    c_pred = (cs >= 0.5).to(torch.long)
    c_true = out_dict["CONCEPTS"].to(torch.long)

    y_pred = out_dict["YS"]
    y_true = out_dict["LABELS"]

    # case for fully neural approach
    if torch.all(cs < 0).item():
        loss = torch.tensor(0.0, device=cs.device)
    else:
        loss = F.binary_cross_entropy(cs, c_true.float())

    cacc = (c_true == c_pred).all(dim=1).float().mean().item()

    # Split predictions into separate tensors for each label
    y_pred_split = torch.split(y_pred, 2, dim=1)
    y_pred_binary = torch.stack([pred.argmax(dim=1) for pred in y_pred_split], dim=1)

    acc = accuracy_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy())
    f1 = f1_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy(), average="micro")

    return loss, cacc * 100, acc * 100, f1 * 100


def compute_clevr_predictions(logits):
    n_colors = 8
    n_shapes = 6
    n_materials = 2
    n_sizes = 3

    concept_meaning = {
        "colors": slice(0, n_colors),
        "shapes": slice(n_colors, n_colors + n_shapes),
        "materials": slice(n_colors + n_shapes, n_colors + n_shapes + n_materials),
        "sizes": slice(
            n_colors + n_shapes + n_materials,
            n_colors + n_shapes + n_materials + n_sizes,
        ),
    }

    preds = []
    for _, value in concept_meaning.items():
        if isinstance(logits, np.ndarray):
            max_index = np.argmax(logits[:, :, value], axis=1)
            one_hot = np.zeros_like(logits[:, :, value])
            one_hot[np.arange(one_hot.shape[0])[:, None], max_index] = 1
        elif isinstance(logits, torch.Tensor):
            max_index = torch.argmax(logits[:, :, value], dim=1)
            one_hot = torch.zeros_like(logits[:, :, value])
            one_hot.scatter_(1, max_index.unsqueeze(1), 1)
        else:
            raise ValueError(
                "Unsupported data type. Please provide either a numpy array or a torch tensor."
            )
        preds.append(one_hot)
    preds = (
        torch.cat(preds, dim=2)
        if isinstance(logits, torch.Tensor)
        else np.concatenate(preds, axis=2)
    )

    return preds


def get_world_probabilities_matrix(
    c_prb_1: ndarray, c_prb_2: ndarray
) -> Tuple[ndarray, ndarray]:
    """Get the world probabilities

    Args:
        c_prb_1 (ndarray): concept probability 1
        c_prb_2 (ndarray): concept probability 2
    NB:
        c_prb_1 is the concept probability associated to the first logit (concept of the first image) of all the images in a batch
            e.g shape (256, 10) where 256 is the batch size and 10 is the cardinality of all possible concepts
        c_prb_2 is the concept probability associated to the second logit (concept of the second image)
    Returns:
        decomposed_world_prob (ndarray): matrix of shape (batch_size, concept_cardinality, concept_cardinality)
        worlds_prob (ndarray): matrix of shape (batch_size, concept_cardinality^2) (where each row is put one after the other)
    """
    # Add dimension to c_prb_1 and c_prb_2
    c_prb_1_expanded = np.expand_dims(c_prb_1, axis=-1)  # 256, 10, 1
    c_prb_2_expanded = np.expand_dims(c_prb_2, axis=1)  # 256, 1, 10

    # Compute the outer product to get c_prbs
    decomposed_world_prob = np.matmul(c_prb_1_expanded, c_prb_2_expanded)  # 256, 10, 10

    # Reshape c_prbs to get worlds_prob
    worlds_prob = decomposed_world_prob.reshape(
        decomposed_world_prob.shape[0], -1
    )  # 256, 100

    # return both
    return decomposed_world_prob, worlds_prob


def get_mean_world_probability(
    decomposed_world_prob: ndarray, c_true_cc: ndarray
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Get the mean world probabilities

    Args:
        decomposed_world_prob (ndarray): decomposed world probability
        c_true_cc (ndarray): concept probability per concept

    Returns:
        mean_world_prob (ndarray): mean world probability
        world_counter (ndarray): world counter
    """
    mean_world_prob = dict()
    world_counter = dict()

    # loop over the grountruth concept of each sample (concept1, concept2)
    for i, (c_1, c_2) in enumerate(c_true_cc):
        world_label = str(c_1) + str(
            c_2
        )  # get the world as a string for the key of the dictionary

        # fill if it is zero
        if world_label not in world_counter:
            mean_world_prob[world_label] = 0
            world_counter[world_label] = 0

        # get that world concept probability
        mean_world_prob[world_label] += decomposed_world_prob[i, c_1, c_2]
        # count that world
        world_counter[world_label] += 1

    # Normalize
    for el in mean_world_prob:
        mean_world_prob[el] = (
            0 if world_counter[el] == 0 else mean_world_prob[el] / world_counter[el]
        )

    return mean_world_prob, world_counter


def get_alpha(
    e_world_counter: ndarray, c_true_cc: ndarray, n_facts=5
) -> Tuple[Dict[str, ndarray], Dict[str, int]]:
    """Get alpha map

    Args:
        e_world_counter (ndarray): de world counter
        c_true_cc (ndarray): groundtruth concepts
        n_facts (int): number of concepts

    Returns:
        mean_world_prob (ndarray): mean world probability
        world_counter (ndarray): world counter
    """

    mean_world_prob = dict()
    world_counter = dict()

    # loop over the grountruth concept of each sample (concept1, concept2)
    for i, (c_1, c_2) in enumerate(c_true_cc):
        world_label = str(c_1) + str(
            c_2
        )  # get the world as a string for the key of the dictionary

        # fill if it is zero
        if world_label not in world_counter:
            mean_world_prob[world_label] = np.zeros(n_facts**2)
            world_counter[world_label] = 0

        # get that world concept probability
        mean_world_prob[world_label] += e_world_counter[i]
        # count that world
        world_counter[world_label] += 1

    # Normalize
    for el in mean_world_prob:
        mean_world_prob[el] = (
            np.zeros(n_facts**2)
            if world_counter[el] == 0
            else mean_world_prob[el] / world_counter[el]
        )

        if world_counter[el] != 0:
            assert (
                np.sum(mean_world_prob[el]) > 0.99
                and np.sum(mean_world_prob[el]) < 1.01
            ), mean_world_prob[el]

    return mean_world_prob, world_counter


def get_alpha_single(
    e_world_counter: ndarray, c_true: ndarray, n_facts=5
) -> Tuple[Dict[str, ndarray], Dict[str, int]]:
    """Get alpha map for single concept

    Args:
        e_world_counter (ndarray): de world counter
        c_true (ndarray): groundtruth concepts
        n_facts (int): number of concepts

    Returns:
        mean_world_prob (ndarray): mean world probability
        world_counter (ndarray): world counter
    """

    mean_world_prob = dict()
    world_counter = dict()

    # loop over the grountruth concept of each sample (concept)
    for i, c in enumerate(c_true):
        world_label = str(c)  # get the world as a string for the key of the dictionary

        # fill if it is zero
        if world_label not in world_counter:
            mean_world_prob[world_label] = np.zeros(n_facts)
            world_counter[world_label] = 0

        # get that world concept probability
        mean_world_prob[world_label] += e_world_counter[i]
        # count that world
        world_counter[world_label] += 1

    # Normalize
    for el in mean_world_prob:
        mean_world_prob[el] = (
            np.zeros(n_facts)
            if world_counter[el] == 0
            else mean_world_prob[el] / world_counter[el]
        )

        if world_counter[el] != 0:
            assert (
                np.sum(mean_world_prob[el]) > 0.99
                and np.sum(mean_world_prob[el]) < 1.01
            ), mean_world_prob[el]

    return mean_world_prob, world_counter


def get_concept_probability(model, loader):
    """Get concept probabilities out of a loader

    Args:
        model (nn.Module): network
        loader: dataloader

    Returns:
        pc (ndarray): probabilities of concepts
        c_prb_1 (ndarray): probabilities of concept 1
        c_prb_2 (ndarray): probabilities of concept 2
        gs (ndarray): groundtruth concepts
    """
    for i, data in enumerate(loader):
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        if i == 0:
            c_prb = out_dict["pCS"].detach().cpu().numpy()
            c_true = concepts.detach().cpu().numpy()
        else:
            c_prb = np.concatenate(
                [c_prb, out_dict["pCS"].detach().cpu().numpy()], axis=0
            )
            c_true = np.concatenate([c_true, concepts.detach().cpu().numpy()], axis=0)

    c_prb_1 = c_prb[:, 0, :]  # [#dati, #facts]
    c_prb_2 = c_prb[:, 1, :]  # [#dati, #facts]

    c_true_1 = c_true[:, 0]  # [#dati, #facts]
    c_true_2 = c_true[:, 1]  # [#dati, #facts]

    gs = np.char.add(c_true_1.astype(str), c_true_2.astype(str))
    gs = gs.astype(int)

    c1 = np.expand_dims(c_prb_1, axis=2)  # [#dati, #facts, 1]
    c2 = np.expand_dims(c_prb_2, axis=1)  # [#dati, 1, #facts]

    pc = np.matmul(c1, c2)

    pc = pc.reshape(c1.shape[0], -1)  # [#dati, #facts^2]

    return pc, c_prb_1, c_prb_2, gs


def calculate_mean_pCs(
    ensemble_c_prb_1: ndarray, ensemble_c_prb_2: ndarray, ensamble_len: int
) -> ndarray:
    """Get concept probabilities for mcdropout

    Args:
        ensemble_c_prb_1 (ndarray): ensemble probability for concept 1
        ensemble_c_prb_2 (ndarray): ensemble probability for concept 2
        ensemble_len (int): dimension of the ensemble

    Returns:
        mean pc (ndarray): average concept probability
    """
    pCs = []

    for i in range(ensamble_len):
        pc1 = np.expand_dims(ensemble_c_prb_1[i], axis=2)
        pc2 = np.expand_dims(ensemble_c_prb_2[i], axis=1)

        pc = np.matmul(pc1, pc2).reshape(pc1.shape[0], -1)
        pCs.append(pc)

    return np.mean(pCs, axis=0)  # (6000,100)


def _bin_initializer(num_bins: int) -> Dict[int, Dict[str, int]]:
    """Initialize bins for ECE computation

    Args:
        num_bins (int): number of bins

    Returns:
        bins: dictioanry containing confidence, accuracy and count for each bin
    """
    # Builds the bin
    return {
        i: {"COUNT": 0, "CONF": 0, "ACC": 0, "BIN_ACC": 0, "BIN_CONF": 0}
        for i in range(num_bins)
    }


def _populate_bins(
    confs: ndarray, preds: ndarray, labels: ndarray, num_bins: int
) -> Dict[int, Dict[str, float]]:
    """Populates the bins for ECE computation

    Args:
        confs (ndarray): confidence
        preds (ndarray): predictions
        labels (ndarray): labels
        num_bins (int): number of bins

    Returns:
        bin_dict: dictionary containing confidence, accuracy and count for each bin
    """
    # initializes n bins (a bin contains probability from x to x + smth (where smth is greater than zero))
    bin_dict = _bin_initializer(num_bins)

    for confidence, prediction, label in zip(confs, preds, labels):
        binn = int(math.ceil(num_bins * confidence - 1))
        if binn == -1:
            binn = 0
        bin_dict[binn]["COUNT"] += 1
        bin_dict[binn]["CONF"] += confidence
        bin_dict[binn]["ACC"] += 1 if label == prediction else 0

    for bin_info in bin_dict.values():
        bin_count = bin_info["COUNT"]
        if bin_count == 0:
            bin_info["BIN_ACC"] = 0
            bin_info["BIN_CONF"] = 0
        else:
            bin_info["BIN_ACC"] = bin_info["ACC"] / bin_count
            bin_info["BIN_CONF"] = bin_info["CONF"] / bin_count

    return bin_dict


def expected_calibration_error(
    confs: ndarray, preds: ndarray, labels: ndarray, num_bins: int = 10
) -> Tuple[float, Dict[str, float]]:
    """Computes the ECE

    Args:
        confs (ndarray): confidence
        preds (ndarray): predictions
        labels (ndarray): labels
        num_bins (int): number of bins

    Returns:
        bin_dict: dictionary containing confidence, accuracy and count for each bin
    """
    # Perfect calibration is achieved when the ECE is zero
    # Formula: ECE = sum 1 upto M of number of elements in bin m|Bm| over number of samples across all bins (n), times |(Accuracy of Bin m Bm) - Confidence of Bin m Bm)|

    bin_dict = _populate_bins(confs, preds, labels, num_bins)  # populate the bins
    num_samples = len(labels)  # number of samples (n)
    ece = sum(
        (bin_info["BIN_ACC"] - bin_info["BIN_CONF"]).__abs__()
        * bin_info["COUNT"]
        / num_samples
        for bin_info in bin_dict.values()  # number of bins basically
    )
    return ece, bin_dict


def expected_calibration_error_by_concept(
    confs: ndarray,
    preds: ndarray,
    labels: ndarray,
    groundtruth: int,
    num_bins: int = 10,
) -> Tuple[float, Dict[str, float]]:
    """Computes the ECE filtering by index

    Args:
        confs (ndarray): confidence
        preds (ndarray): predictions
        labels (ndarray): labels
        groundtruth (int): groundtruth value
        num_bins (int): number of bins

    Returns:
        ece: ece value
        bin_dict: dictionary containing confidence, accuracy and count for each bin
    """
    indices = np.where(labels == groundtruth)[0]
    if np.size(indices) > 0:
        return expected_calibration_error(
            confs[indices], preds[indices], labels[indices], num_bins
        )
    return None


def entropy(probabilities: ndarray, n_values: int):
    """Entropy

    Args:
        probabilities (ndarray): probability vector
        n_values (int): n values

    Returns:
        entropy_values: entropy vector
    """
    # Compute entropy along the columns
    probabilities += 1e-5
    probabilities /= 1 + (n_values * 1e-5)

    entropy_values = -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(
        n_values
    )
    return entropy_values


def mean_entropy(probabilities: ndarray, n_values: int) -> float:
    """Mean Entropy

    Args:
        probabilities (ndarray): probability vector
        n_values (int): n values

    Returns:
        entropy_values: mean entropy
    """
    # Accepts a ndarray of dim n_examples * n_classes (or equivalently n_concepts)
    entropy_values = entropy(probabilities, n_values).mean()
    return entropy_values.item()


def variance(probabilities: np.ndarray, n_values: int):
    """Variance

    Args:
        probabilities (ndarray): probability vector
        n_values (int): n values

    Returns:
        variance_values: variance
    """
    # Compute variance along the columns
    mean_values = np.mean(probabilities, axis=1, keepdims=True)
    # Var(X) = E[(X - mu)^2] c= 1/n-1 (X - mu)**2 (unbiased estimator)
    variance_values = np.sum((probabilities - mean_values) ** 2, axis=1) / (
        n_values - 1
    )
    return variance_values


def mean_variance(probabilities: np.ndarray, n_values: int) -> float:
    """Mean Variance

    Args:
        probabilities (ndarray): probability vector
        n_values (int): n values

    Returns:
        variance_values: mean variance
    """
    # Accepts a ndarray of dim n_examples * n_classes (or equivalently n_concepts)
    variance_values = variance(probabilities, n_values).mean()
    return variance_values.item()


def class_mean_entropy(
    probabilities: ndarray, true_classes: ndarray, n_classes: int
) -> ndarray:
    """Function which computes a mean entropy per class

    Args:
        probabilities (ndarray): probability vector
        true_classes (ndarray): grountruth classes
        n_classes (int): number of classes

    Returns:
        class_mean_entropy_values: mean entropy per class
    """
    # Compute the mean entropy per class
    num_samples, num_classes = probabilities.shape

    class_mean_entropy_values = np.zeros(
        n_classes
    )  # all possible results by summing 2 digits
    class_counts = np.zeros(n_classes)

    for i in range(num_samples):
        sample_entropy = entropy(np.expand_dims(probabilities[i], axis=0), num_classes)
        class_mean_entropy_values[true_classes[i]] += sample_entropy
        class_counts[true_classes[i]] += 1

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    class_mean_entropy_values /= class_counts

    return class_mean_entropy_values


def class_mean_variance(
    probabilities: ndarray, true_classes: ndarray, n_classes: int
) -> ndarray:
    """Function which computes the class mean variance

    Args:
        probabilities (ndarray): probability vector
        true_classes (ndarray): grountruth classes
        n_classes (int): number of classes

    Returns:
        class_mean_variance: mean variance per class
    """
    # Compute the mean variance per class
    num_samples, num_classes = probabilities.shape

    class_mean_variance_values = np.zeros(
        n_classes
    )  # all possible results by summing 2 digits
    class_counts = np.zeros(n_classes)

    for i in range(num_samples):
        sample_variance = variance(
            np.expand_dims(probabilities[i], axis=0), num_classes
        )
        class_mean_variance_values[true_classes[i]] += sample_variance
        class_counts[true_classes[i]] += 1

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    class_mean_variance_values /= class_counts

    return class_mean_variance_values


def mean_l2_distance(vectors: List[ndarray]):
    """Function which computes the p(c|x) distance of a vector

    Args:
        vectors (List[ndarray]): vectors of parameters

    Returns:
        mean dist: mean L2 distance
    """
    num_vectors = len(vectors)
    distances = []

    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):  # exclude itself
            distance = np.linalg.norm(vectors[i] - vectors[j])
            distances.append(distance)

    mean_distance = np.mean(distances)
    return mean_distance


def vector_to_parameters(vec: torch.Tensor, parameters) -> None:
    """Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        None: This function does not return a value.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def get_accuracy_and_counter(
    n_concepts: int, c_pred: ndarray, c_true: ndarray, map_index: bool = False
):
    """Function which counts the occurrece and accuracy

    Args:
        n_concepts (int): number of concepts
        c_pred (ndarray): predicted concepts
        c_true (ndarray): groundtruth concepts
        map_index (bool, default=False): map index

    Returns:
        concept_counter_list: concept counter
        concept_acc_list: concept accuracy
    """
    concept_counter_list = np.zeros(n_concepts)
    concept_acc_list = np.zeros(n_concepts)

    for i, c in enumerate(c_true):
        import math

        # From module n_concept to module 10
        if map_index:
            # get decimal and unit
            decimal = c // 10
            unit = c % 10
            # print("C remapped: from", c, "to", unit + decimal*n_concepts, "decimal", decimal, "unit", unit, "nconc", int(math.sqrt(n_concepts)))
            c = unit + decimal * int(math.sqrt(n_concepts))

        concept_counter_list[c] += 1

        if c == c_pred[i]:
            concept_acc_list[c] += 1

    for i in range(len(concept_counter_list)):
        if concept_counter_list[i] != 0:
            concept_acc_list[i] /= concept_counter_list[i]

    return concept_counter_list, concept_acc_list


def concept_accuracy(c1_prob: ndarray, c2_prob: ndarray, c_true: ndarray):
    """Function which computes the concept accuracy

    Args:
        c1_prob (ndarray): first concept probability
        c2_prob (ndarray): second concept probability
        c_true (ndarray): groundtruth concepts

    Returns:
        concept_counter_list: concept counter
        concept_acc_list: concept accuracy
    """
    n_concepts = c1_prob.shape[1]

    c_pred_1 = np.argmax(c1_prob, axis=1)
    c_pred_2 = np.argmax(c2_prob, axis=1)

    c_pred = np.concatenate((c_pred_1, c_pred_2), axis=0)

    return get_accuracy_and_counter(n_concepts, c_pred, c_true)


def world_accuracy(world_prob: ndarray, world_true: ndarray, n_concepts: int):
    """Function which computes the world accuracy

    Args:
        world_prob (ndarray): world probability
        world_true (ndarray): groundtruth world concepts
        n_concepts (int): number of concepts

    Returns:
        concept_counter_list: world concept counter
        concept_acc_list: world concept accuracy
    """
    n_world = world_prob.shape[1]
    world_pred = np.argmax(world_prob, axis=1)

    decimal_values = np.array([x // n_concepts for x in world_pred])
    unit_values = np.array([x % n_concepts for x in world_pred])

    world_pred = np.array(
        np.char.add(decimal_values.astype(str), unit_values.astype(str))
    ).astype(int)

    return get_accuracy_and_counter(n_world, world_pred, world_true, True)
