# Module which logs stuff on wandb

import wandb


def wandb_log_step(i: int, epoch: int, loss, losses=None):
    """Wandb log loss step
    Args:
        i (int): iteration number
        epoch (int): iteration number
        loss: loss value
        losses (default=None): dictionary concerning addictional loss functions

    Returns:
        None: This function does not return a value.
    """
    wandb.log({"loss": loss, "epoch": epoch, "step": i})
    if losses is not None:
        wandb.log(losses)


def wandb_log_epoch(**kwargs):
    """Wandb log epoch metrics: accuracy on labels, accuracy on concepts, learning rate, test loss
    Args:
        kwargs: key-value dictionary with metrics

    Returns:
        None: This function does not return a value.
    """
    # log accuracies
    epoch = kwargs["epoch"]
    acc = kwargs["acc"]
    c_acc = kwargs["cacc"]
    wandb.log({"acc": acc, "c-acc": c_acc, "epoch": epoch})

    lr = kwargs["lr"]
    wandb.log({"lr": lr})

    tloss = kwargs["tloss"]
    wandb.log({"test-loss": tloss})


def wand_log_end(t_acc, t_c_acc):
    """Wandb log epoch metrics on the test set: accuracy on labels and on concepts
    Args:
        kwargs: key-value dictionary with metrics

    Returns:
        None: This function does not return a value.
    """
    # log score metrics
    wandb.log({"test-acc": t_acc, "test-c_acc": t_c_acc})


def wandb_log_step_prefix(prefix, i, epoch, loss, losses=None):
    """Log losses on wandb with prefix

    Args:
        prefix (str): prefix for the log
        i (int): iteration
        epoch (int): epoch
        loss: loss value
        losses (default=None): dictionary for additional loss functions

    Returns:
        None: This function does not return a value.
    """
    wandb.log({f"{prefix}_loss": loss, f"{prefix}_epoch": epoch, f"{prefix}_step": i})
    if losses is not None:
        wandb.log(losses)


def wandb_log_epoch_prefix(prefix, **kwargs):
    """Log epochs metrix on wandb with prefix

    Args:
        prefix (str): prefix for the log
        i (int): iteration
        epoch (int): epoch
        loss: loss value
        losses (default=None): dictionary for additional loss functions

    Returns:
        None: This function does not return a value.
    """
    # log accuracies
    epoch = kwargs["epoch"]
    acc = kwargs["acc"]
    c_acc = kwargs["cacc"]
    wandb.log(
        {f"{prefix}_acc": acc, f"{prefix}_c-acc": c_acc, f"{prefix}_epoch": epoch}
    )

    lr = kwargs["lr"]
    wandb.log({f"{prefix}_lr": lr})

    tloss = kwargs["tloss"]
    wandb.log({f"{prefix}_test-loss": tloss})
