import numpy as np
from utils.wandb_logger import *
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL
from utils.metrics import evaluate_metrics
from utils import fprint

import torch
import numpy as np


def test(model: MnistDPL, dataset: BaseDataset, args):
    """TRAINING

    Args:
        model (MnistDPL): network
        dataset (BaseDataset): dataset Kandinksy
        _loss (ADDMNIST_DPL): loss function
        args: parsed args

    Returns:
        None: This function does not return a value.
    """

    # best f1
    seeds = [123, 456, 789, 1011, 1213]
    caccs, yaccs, f1s, cf1s = [], [], [], []

    for seed in seeds:
        current_model_path = f"best_model_{args.dataset}_{args.model}_{seed}.pth"

        # Default Setting for Training
        model.to(model.device)
        # retrieve the status dict
        model_state_dict = torch.load(current_model_path)
        # Load the model status dict
        model.load_state_dict(model_state_dict)
        model.eval()
        _, _, test_loader = dataset.get_data_loaders()

        fprint("\n--- Start of Evaluation ---\n")

        tloss, cacc, yacc, f1, cf1 = evaluate_metrics(
            model, test_loader, args, cf1=True
        )
        caccs.append(cacc)
        yaccs.append(yacc)
        f1s.append(f1)
        cf1s.append(cf1)

        print("Concept Acc", cacc, "Label Acc", yacc, "Label F1", f1, "Concept F1", cf1)

    print("Concept Acc", np.mean(caccs), "pm", np.std(caccs))
    print("Label Acc", np.mean(yaccs), "pm", np.std(yaccs))
    print("F1 Label", np.mean(f1s), "pm", np.std(f1s))
    print("F1 Concept", np.mean(cf1s), "pm", np.std(cf1s))
