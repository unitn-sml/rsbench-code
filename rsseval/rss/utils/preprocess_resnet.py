# This module contains the preoprocessing operation for Kandisnky using a ResNet

import numpy as np

import os

from utils.wandb_logger import *
from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset
from models.mnistdpl import MnistDPL
from utils.dpl_loss import ADDMNIST_DPL

from utils import fprint


def preprocess(model: MnistDPL, dataset: BaseDataset, args):
    """Preprocess Kandinksy images
    Args:
        model: network
        dataset: dataset
        args: command line arguments

    Returns:
        None: This function does not return a value.
    """
    # Default Setting for Training
    model.to(model.device)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    fprint("\n--- Start of Preprocessing ---\n")

    model.eval()

    print("Doing training")

    os.makedirs("data/kand-preprocess/train/images/", exist_ok=True)
    os.makedirs("data/kand-preprocess/train/labels/", exist_ok=True)
    os.makedirs("data/kand-preprocess/train/concepts/", exist_ok=True)

    for i, data in enumerate(train_loader):
        id, images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        emb = out_dict["EMBS"]

        id = id.item()
        emb = emb.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        g = concepts.detach().cpu().numpy()

        np.save(f"data/kand-preprocess/train/images/{str(id).zfill(5)}", emb)
        np.save(f"data/kand-preprocess/train/labels/{str(id).zfill(5)}", y)
        np.save(f"data/kand-preprocess/train/concepts/{str(id).zfill(5)}", g)

        if i % 10 == 0:
            progress_bar(i, len(train_loader) - 9, 0, 0)

    print("Doing validation")

    os.makedirs("data/kand-preprocess/val/images/", exist_ok=True)
    os.makedirs("data/kand-preprocess/val/labels/", exist_ok=True)
    os.makedirs("data/kand-preprocess/val/concepts/", exist_ok=True)

    for i, data in enumerate(val_loader):
        id, images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        emb = out_dict["EMBS"]

        id = id.item()
        emb = emb.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        g = concepts.detach().cpu().numpy()

        np.save(f"data/kand-preprocess/val/images/{str(id).zfill(5)}", emb)
        np.save(f"data/kand-preprocess/val/labels/{str(id).zfill(5)}", y)
        np.save(f"data/kand-preprocess/val/concepts/{str(id).zfill(5)}", g)

        if i % 10 == 0:
            progress_bar(i, len(val_loader) - 9, 0, 0)

    print("Doing testing")

    os.makedirs("data/kand-preprocess/test/images/", exist_ok=True)
    os.makedirs("data/kand-preprocess/test/labels/", exist_ok=True)
    os.makedirs("data/kand-preprocess/test/concepts/", exist_ok=True)

    for i, data in enumerate(test_loader):
        id, images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )

        out_dict = model(images)

        emb = out_dict["EMBS"]

        id = id.item()
        emb = emb.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        g = concepts.detach().cpu().numpy()

        np.save(f"data/kand-preprocess/test/images/{str(id).zfill(5)}", emb)
        np.save(f"data/kand-preprocess/test/labels/{str(id).zfill(5)}", y)
        np.save(f"data/kand-preprocess/test/concepts/{str(id).zfill(5)}", g)

        if i % 10 == 0:
            progress_bar(i, len(test_loader) - 9, 0, 0)
    # ids  = np.concatenate(ids,  dim=0).detach().cpu().numpy()
    # embs = np.concatenate(embs, dim=0).detach().cpu().numpy()
    # ys   = np.concatenate(ys,   dim=0).detach().cpu().numpy()
    # gs   = np.concatenate(gs,   dim=0).detach().cpu().numpy(

    return
