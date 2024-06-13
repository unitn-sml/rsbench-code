# Module which contains the KL of two Gaussian distributions


def kl_divergence(mu, logsigma, reduction="mean"):
    """KL between two normal distributions
    Args:
        mu: mean of the Gaussian
        logsigma: std of the Gaussian
        reduction: which reduction to use

    Returns:
        kl: KL divergence value
    """
    kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
    if reduction == "sum":
        return kl.sum()
    else:
        return kl.sum(dim=-1).mean()
