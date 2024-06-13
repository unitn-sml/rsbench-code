import torch
import torch.nn as nn


"""
class BOIAConceptizer:
    network definitions of encoder and decoder using fully connected network
    encoder c() is the network by computing concept c(e(x))
def __init__:
    define parameters (e.g., # of layers) 
def encode:
    compute concepts
"""


class BOIAConceptizer(nn.Module):
    """
    def __init__:
        define parameters (e.g., # of layers)
        MLP-based conceptizer for concept basis learning.
    Inputs:
        din (int): input size
        nconcept (int): # of all concepts
    Return:
        None
    """

    def __init__(self, din, nconcept):
        super(BOIAConceptizer, self).__init__()

        # set self hyperparameters
        self.din = din  # Input dimension
        self.nconcept = nconcept  # Number of "atoms"/concepts

        """
        encoding
        self.enc1: encoder for known concepts
        """
        self.enc1 = nn.Linear(self.din, self.nconcept)

    """ 
    def forward:
        compute concepts
    Inputs:
        x: output of pretrained model (encoder)
    Return:
        encoded_1: predicted known concepts
    """

    def forward(self, x):

        # resize
        p = x.view(x.size(0), -1)

        T = 2.5
        logits_c = self.enc1(p) / T
        encoded_1 = torch.sigmoid(logits_c)

        return encoded_1
