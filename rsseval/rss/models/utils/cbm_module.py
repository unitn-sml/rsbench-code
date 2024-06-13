from torch import nn
from utils.conf import get_device


class CBMModule(nn.Module):
    def __init__(self, encoder, model_dict=None, n_facts=20, nr_classes=19):
        super(CBMModule, self).__init__()
        self.encoder = encoder
        self.model_dict = model_dict  # Dictionary of pre-compiled ProbLog models
        self.device = get_device()
        self.nr_classes = nr_classes
        self.n_facts = n_facts

    def forward(self, x):
        z = self.encoder(x)
        # normalize concept preditions
        self.facts_probs = self.normalize_concepts(z)
        # inference
        self.query_prob = self.inference(self.facts_probs)

        return self.query_prob, self.facts_probs

    def inference(self, facts_probs):
        """Returns the output probability of the model with the classifier"""
        pass

    def normalize_concepts(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        pass
