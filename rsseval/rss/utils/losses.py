# Losses module
import torch
import numpy as np
import torch.nn.functional as F


def ADDMNIST_Classification(out_dict: dict, args):
    """Addmnist classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    if args.model in [
        "mnistdpl",
        "mnistdplrec",
        "mnistpcbmdpl",
        "mnistclip",
        "mnistnn",
        "mnistcbm",
    ]:
        loss = F.nll_loss(out.log(), labels, reduction="mean")
    elif args.model in [
        "mnistsl",
        "mnistslrec",
        "mnistpcbmsl",
    ]:
        loss = F.cross_entropy(out, labels, reduction="mean")
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def ADDMNIST_Concept_Match(out_dict: dict, args):
    """Addmnist concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["CS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)
    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)
    loss = torch.tensor(0.0, device=reprs.device)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    for j in range(len(objs)):
        mask = g_objs[j] != -1
        if mask.sum() > 0:
            loss += torch.nn.CrossEntropyLoss()(
                objs[j][mask].squeeze(1), g_objs[j][mask].view(-1)
            )
    losses = {"c-loss": loss.item()}

    print(loss.item() / len(objs))

    return loss / len(objs), losses


def ADDMNIST_Concept_CLIP(out_dict: dict, args):
    """Addmnist concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"]
    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)
    loss = torch.tensor(0.0, device=reprs.device)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    for j in range(len(objs)):

        input_prob = objs[j].squeeze(1)
        targt_prob = g_objs[j].squeeze(1)

        loss += F.kl_div(input_prob.log(), targt_prob)
    losses = {"clip-loss": loss.item()}

    return loss / len(objs), losses


def ADDMNIST_REC_Match(out_dict: dict, args):
    """Addmnist concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    recs, inputs, mus, logvars = (
        out_dict["RECS"],
        out_dict["INPUTS"],
        out_dict["MUS"],
        out_dict["LOGVARS"],
    )

    L = len(recs)

    assert inputs.size() == recs.size(), f"{len(inputs)}-{len(recs)}"

    recon = F.binary_cross_entropy(recs.view(L, -1), inputs.view(L, -1))
    kld = (-0.5 * (1 + logvars - mus**2 - logvars.exp()).sum(1).mean() - 1).abs()

    losses = {"recon-loss": recon.item(), "kld": kld.item()}

    return recon + args.beta * kld, losses


def ADDMNIST_Entropy(out_dict, args):
    """Addmnist entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    l = pCs.size(-1)
    p_mean = torch.mean(pCs, dim=0).view(-1, l)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=1, keepdim=True)
    p_mean /= Z

    loss = 0
    for i in range(p_mean.size(0)):
        loss -= torch.sum(p_mean[i] * p_mean[i].log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def ADDMNIST_rec_class(out_dict: dict, args):
    """Addmnist rec class

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss1, losses1 = ADDMNIST_Classification(out_dict, args)
    loss2, losses2 = ADDMNIST_REC_Match(out_dict, args)

    losses1.update(losses2)

    return loss1 + args.gamma * loss2, losses1


def ADDMNIST_Cumulative(out_dict: dict, args):
    """Addmnist cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = ADDMNIST_Classification(out_dict, args)
    mitigation = 0
    if args.model in ["mnistdplrec", "mnistslrec", "mnistltnrec"]:
        loss1, losses1 = ADDMNIST_REC_Match(out_dict, args)
        mitigation += args.w_rec * loss1
        losses.update(losses1)
    if args.entropy:
        loss2, losses2 = ADDMNIST_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = ADDMNIST_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)
    # if args.dataset in ['clipshortmnist']:
    #     loss4, losses4 = ADDMNIST_Concept_CLIP(out_dict, args)
    #     mitigation += loss4
    #     losses.update(losses4)

    return loss + args.gamma * mitigation, losses


def KAND_Classification(out_dict: dict, args):
    """Kandinsky classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    out = out_dict["YS"]
    # preds = out_dict["PREDS"]
    final_labels = out_dict["LABELS"][:, -1].to(torch.long)
    # inter_labels = out_dict["LABELS"][:, :-1].to(torch.long)

    if args.task in ["patterns"]:
        weight = torch.tensor(
            [
                1 / 0.04938272,
                1 / 0.14814815,
                1 / 0.02469136,
                1 / 0.14814815,
                1 / 0.44444444,
                1 / 0.07407407,
                1 / 0.02469136,
                1 / 0.07407407,
                1 / 0.01234568,
            ],
            device=out.device,
        )
        final_weight = torch.tensor([0.5, 0.5], device=out.device)
    elif args.task == "red_triangle":
        weight = torch.tensor([0.35538, 1 - 0.35538], device=out.device)
        final_weight = torch.tensor([0.04685, 1 - 0.04685], device=out.device)
    else:
        weight = torch.tensor([0.5, 0.5], device=out.device)
        final_weight = torch.tensor([0.5, 0.5], device=out.device, dtype=torch.float64)

    if args.model in [
        "kanddpl",
        "kandcbm",
        "kandnn",
        "kandclip",
        "minikanddpl",
        "kandcbm",
    ]:
        ## ADD SMALL OFFSET
        # out += 1e-5
        # with torch.no_grad():
        #     Z = torch.sum(out, dim=1, keepdim=True)
        # out /= Z

        if args.model in ["kandcbm"]:
            criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=final_weight.float()
            )
            loss = criterion(out, final_labels)
        else:
            loss = F.nll_loss(
                out.log(), final_labels, reduction="mean"  # , weight=final_weight
            )
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, f"{loss}, {out}, {final_labels}"

    losses = {"y-loss": loss.item()}

    return loss, losses


def KAND_Concept_Match(out_dict: dict):
    """Kandinsky concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["CS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    objs = torch.split(reprs, 1, dim=1)
    g_objs = torch.split(concepts, 1, dim=1)

    loss = torch.tensor(0.0, device=reprs.device)

    assert len(objs) == len(g_objs), f"{len(objs)}-{len(g_objs)}"

    for j in range(len(g_objs)):

        # Loop though the figures

        cs = torch.split(objs[j], 3, dim=-1)
        gs = torch.split(g_objs[j], 1, dim=-1)

        assert len(cs) == len(gs), f"{len(cs)}-{len(gs)}"

        for k in range(len(gs)):
            target = gs[k].view(-1)
            mask = target != -1
            if mask.sum() > 0:
                loss += torch.nn.CrossEntropyLoss()(
                    cs[k][mask].squeeze(1), target[mask].view(-1)
                )

    loss /= len(g_objs) * len(gs)

    losses = {"c-loss": loss.item()}

    return loss, losses


def KAND_Entropy(out_dict, args):
    """Kandinsky entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]

    pc_i = torch.cat(torch.split(pCs, 3, dim=-1), dim=1)

    p_mean = torch.mean(pc_i, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=1, keepdim=True)
    p_mean /= Z

    loss = 0
    for i in range(p_mean.size(0)):
        loss -= torch.sum(p_mean[i] * p_mean[i].log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > 0, loss

    return 1 - loss, losses


def KAND_Cumulative(out_dict: dict, args):
    """Kandinsky cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = KAND_Classification(out_dict, args)

    mitigation = 0
    if args.model in ["kandplrec", "kandslrec", "kandltnrec"]:
        return NotImplementedError("not available")
    if args.entropy:
        loss2, losses2 = KAND_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = KAND_Concept_Match(out_dict)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    # return mitigation, losses
    return loss + args.gamma * mitigation, losses


def SDDOIA_Classification(out_dict: dict, args):
    """SDDOIA classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = 0, {}

    if args.boia_model in ["ce"]:
        loss, losses1 = SDDOIA_CE(out_dict, args)
        losses.update(losses1)
    elif args.boia_model in ["bce"]:
        loss, losses1 = SDDOIA_BCE(out_dict, args)
        losses.update(losses1)
    else:
        raise NotImplementedError("Not implemented loss")

    return loss, losses


def SDDOIA_Entropy(out_dict, args):
    """SDDOIA entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    p_mean = torch.mean(pCs, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=0, keepdim=True)
    p_mean /= Z

    loss = -torch.sum(p_mean * p_mean.log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def SDDOIA_Concept_Match(out_dict: dict, args):
    """SDDOIA concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    loss = torch.tensor(0.0, device=reprs.device)

    probs_list = torch.split(reprs, 2, dim=1)

    for i, rep in enumerate(probs_list):
        # Create a mask to filter out concepts with -1
        mask = concepts[:, i] != -1
        if mask.sum() > 0:  # Proceed only if there are valid entries
            # Apply the mask
            filtered_rep = rep[mask]
            filtered_concepts = concepts[mask, i]
            loss += torch.nn.NLLLoss()(filtered_rep.log(), filtered_concepts)

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses


def SDDOIA_Cumulative(out_dict: dict, args):
    """SDDOIA cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = SDDOIA_Classification(out_dict, args)

    mitigation = 0
    if args.entropy:
        loss2, losses2 = SDDOIA_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = SDDOIA_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses


def SDDOIA_BCE(out_dict: dict, args):
    """SDDOIA bce

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """

    def BCE_forloop(tar, pred):
        loss = F.binary_cross_entropy(tar[0, :4], pred[0, :4])

        for i in range(1, len(tar)):
            loss = loss + F.binary_cross_entropy(tar[i, :4], pred[i, :4])
        return loss

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    loss = BCE_forloop(out, labels)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def SDDOIA_CE(out_dict: dict, args):
    """SDDOIA bce

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """

    def CE_forloop(y_pred, y_true):

        y_trues = torch.split(y_true, 1, dim=-1)
        y_preds = torch.split(y_pred, 2, dim=-1)

        loss = 0
        for i in range(4):

            true = y_trues[i].view(-1)
            pred = y_preds[i]

            ## add small offset to avoid NaNs
            pred = pred + 1e-5
            with torch.no_grad():
                Z = torch.sum(pred, dim=0, keepdim=True)
            pred = pred / Z

            assert torch.max(pred) < 1, pred
            assert torch.min(pred) > 0, pred

            loss_i = F.nll_loss(pred.log(), true.to(torch.long))
            loss += loss_i / 4

            assert loss_i > 0, pred.log()

        return loss

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)

    loss = CE_forloop(out, labels)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses

def XOR_Classification(out_dict: dict, args):
    """XOR classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = 0, {}

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.long)
    
    if args.model in [
        "xorsl",
        "xorcbm",
        "xornn",
        "xordpl",
    ]:
        loss = F.cross_entropy(out, labels, reduction="mean")
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def XOR_Entropy(out_dict, args):
    """XOR entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    p_mean = torch.mean(pCs, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=0, keepdim=True)
    p_mean /= Z

    loss = -torch.sum(p_mean * p_mean.log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def XOR_Concept_Match(out_dict: dict, args):
    """XOR concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)

    loss = torch.tensor(0.0, device=reprs.device)

    for i, rep in enumerate(range(reprs.size(1))):
        # Create a mask to filter out concepts with -1
        filtered_rep = reprs[:, i]
        filtered_concepts = concepts[:, i]
        loss += torch.nn.NLLLoss()(filtered_rep.log(), filtered_concepts)

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses

def XOR_Cumulative(out_dict: dict, args):
    """Xor cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = XOR_Classification(out_dict, args)

    mitigation = 0
    if args.entropy:
        loss2, losses2 = XOR_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = XOR_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses

def MNMATH_Classification(out_dict: dict, args):
    """XOR classification loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = 0, {}

    out = out_dict["YS"]
    labels = out_dict["LABELS"].to(torch.float)

    if args.model in [
        "mnmathnn",
        "mnmathcbm",
        "mnmathsl",
        "mnmathdpl",
    ]:
        loss += torch.nn.BCELoss()(out, labels)
    else:
        loss = torch.tensor(1e-5)

    assert loss > 0, loss

    losses = {"y-loss": loss.item()}

    return loss, losses


def MNMATH_Entropy(out_dict, args):
    """XOR entropy loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    pCs = out_dict["pCS"]
    p_mean = torch.mean(pCs, dim=0)

    ## ADD SMALL OFFSET
    p_mean += 1e-5

    with torch.no_grad():
        Z = torch.sum(p_mean, dim=0, keepdim=True)
    p_mean /= Z

    loss = -torch.sum(p_mean * p_mean.log()) / np.log(10) / p_mean.size(0)

    losses = {"H-loss": 1 - loss}

    assert (1 - loss) > -0.00001, loss

    return 1 - loss, losses


def MNMATH_Concept_Match(out_dict: dict, args):
    """XOR concept match loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    reprs = out_dict["pCS"]
    concepts = out_dict["CONCEPTS"].to(torch.long)
    concepts = concepts.view(concepts.size(0), concepts.size(1) * concepts.size(2), 1)

    loss = torch.tensor(0.0, device=reprs.device)

    for i, rep in enumerate(range(reprs.size(1))):
        # Create a mask to filter out concepts with -1
        filtered_rep = reprs[:, i]
        filtered_concepts = concepts[:, i].squeeze(1)

        specific_concepts = [0, 5, 9]
        mask = torch.isin(filtered_concepts, torch.tensor(specific_concepts).to(filtered_concepts.device)).to(filtered_concepts.device)

        filtered_concepts = filtered_concepts[mask]
        filtered_predictions = filtered_rep[mask]

        if filtered_concepts.size(0) > 0:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(filtered_predictions, filtered_concepts)

    print("Concept supervision loss", loss.item())

    losses = {"c-loss": loss.item()}

    return loss, losses

def MNMATH_Cumulative(out_dict: dict, args):
    """Xor cumulative loss

    Args:
        out_dict: output dictionary
        args: command line arguments

    Returns:
        loss: loss value
        losses: losses dictionary
    """
    loss, losses = MNMATH_Classification(out_dict, args)

    mitigation = 0
    if args.entropy:
        loss2, losses2 = MNMATH_Entropy(out_dict, args)
        mitigation += args.w_h * loss2
        losses.update(losses2)
    if args.c_sup > 0:
        loss3, losses3 = MNMATH_Concept_Match(out_dict, args)
        mitigation += args.w_c * loss3
        losses.update(losses3)

    return loss + args.gamma * mitigation, losses