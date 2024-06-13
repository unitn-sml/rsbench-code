# Checkpoint module
import torch
import os
from utils.conf import create_path


def _get_tag(args):
    """Get tag for the model name

    Args:
        args: command line arguments

    Returns:
        tag (str): tag for the model name
    """
    tag = "dis" if not args.joint else "joint"
    if args.task == "product" and args.model in ["mnistsl", "mnistslrec"]:
        tag = tag + "-prod"
    if args.task == "multiop":
        tag = tag + "-multiop"
    return tag


def create_load_ckpt(model, args):
    """Method which creates checkpoint if it does not exists and loads it afterwards

    Args:
        model (nn.Module): model
        args: command line arguments

    Returns:
        model (nn.Module): model
    """
    create_path("data/runs")
    create_path("data/ckpts")

    tag = _get_tag(args)

    PATH = f"data/runs/{args.dataset}-{args.model}-{tag}-start.pt"

    if args.checkin is not None:
        model.load_state_dict(torch.load(args.checkin))
    elif os.path.exists(PATH):
        print("Loaded", PATH, "\n")
        model.load_state_dict(torch.load(PATH))
    else:
        print("Created", PATH, "\n")
        torch.save(model.state_dict(), PATH)

    return model


def save_model(model, args):
    """Save model in checkpoints

    Args:
        model (nn.Module): model
        args: command line arguments

    Returns:
        None: This function does not return a value.
    """
    create_path("data/ckpts")
    tag = _get_tag(args)

    PATH = f"data/ckpts/{args.dataset}-{args.model}-{tag}-{args.seed}-end.pt"

    if args.checkout:
        print("Saved", PATH, "\n")
        torch.save(model.state_dict(), PATH)


def get_model_name(args):
    """Returns the model name used for saving the checkpoints and dumps

    Args:
        args: command line arguments

    Returns:
        name (str): name of the model
    """
    return f"dset_{args.dataset}-model_{args.model}-tag_{_get_tag(args)}"


def load_checkpoint(model, args, checkin=None):
    """Loads the model from the checkpoint

    Args:
        model (nn.Module): network
        args: command line arguments
        checkin (bool, checkin=None): checkin

    Returns:
        model (nn.Module): model
    """
    create_path("data/ckpts")
    tag = _get_tag(args)

    if checkin is not None:
        PATH = checkin
    else:
        PATH = f"data/ckpts/{args.dataset}-{args.model}-{tag}-{args.seed}-end.pt"

    if not os.path.exists(PATH):
        raise ValueError(f"You have to train the model first, missing {PATH}")

    print("Loaded", PATH, "\n")
    print("Path", PATH)
    model.load_state_dict(torch.load(PATH))

    return model
