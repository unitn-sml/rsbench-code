# DPL loss module

import torch
from utils.normal_kl_divergence import kl_divergence


class ADDMNIST_DPL(torch.nn.Module):
    """Addminst DPL loss class"""

    def __init__(self, loss, nr_classes=19, pcbm=False) -> None:
        """Initialization method

        Args:
            self: instance
            loss: loss value
            nr_classes (int, default=19): number of output classes
            pcbm (bool, default=False): whether to use pcbm or not

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.base_loss = loss
        self.nr_classes = nr_classes
        self.pcbm = pcbm
        self.beta = 0.001

    def forward(self, out_dict, args):
        """Forward

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
            losses: losses dictionary
        """
        loss, losses = self.base_loss(out_dict, args)

        if self.pcbm:
            kl_div = 0

            mus = out_dict["MUS"]
            logvars = out_dict["LOGVARS"]
            for i in range(2):
                kl_div += kl_divergence(mus[i], logvars[i])

            loss += self.beta * kl_div
            losses.update({"kl-div": kl_div})

        return loss, losses


class KAND_DPL(torch.nn.Module):
    """Kandinksy DPL loss"""

    def __init__(self, loss, nr_classes=2) -> None:
        """Initialize method

        Args:
            self: instance
            loss: loss function
            nr_classes: number of classes

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.base_loss = loss
        self.nr_classes = nr_classes

    def forward(self, out_dict, args):
        """Forward method

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
            losses: losses dictionary
        """
        loss, losses = self.base_loss(out_dict, args)
        return loss, losses


class SDDOIA_DPL(torch.nn.Module):
    """SDDOIA DPL loss"""

    def __init__(self, loss, nr_classes=4) -> None:
        """Initialize method

        Args:
            self: instance
            loss: loss function
            nr_classes: number of classes

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.base_loss = loss
        self.nr_classes = nr_classes

    def forward(self, out_dict, args):
        """Forward method

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
            losses: losses dictionary
        """
        loss, losses = self.base_loss(out_dict, args)
        return loss, losses
    

class XOR_DPL(torch.nn.Module):
    """XOR DPL loss"""

    def __init__(self, loss, nr_classes=4) -> None:
        """Initialize method

        Args:
            self: instance
            loss: loss function
            nr_classes: number of classes

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.base_loss = loss
        self.nr_classes = nr_classes

    def forward(self, out_dict, args):
        """Forward method

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
            losses: losses dictionary
        """
        loss, losses = self.base_loss(out_dict, args)
        return loss, losses

class MNMATH_DPL(torch.nn.Module):
    """XOR DPL loss"""

    def __init__(self, loss, nr_classes=2) -> None:
        """Initialize method

        Args:
            self: instance
            loss: loss function
            nr_classes: number of classes

        Returns:
            None: This function does not return a value.
        """
        super().__init__()
        self.base_loss = loss
        self.nr_classes = nr_classes

    def forward(self, out_dict, args):
        """Forward method

        Args:
            self: instance
            out_dict: output dictionary
            args: command line arguments

        Returns:
            loss: loss value
            losses: losses dictionary
        """
        loss, losses = self.base_loss(out_dict, args)
        return loss, losses