import os
from typing import Any, Callable, Optional, Union
import torch
from torchvision import datasets, transforms, models
from torchvision.datasets.folder import default_loader

import preprocessing.clip as clip
from pytorchcv.model_provider import get_model as ptcv_get_model

from preprocessing.mnist_utils import SHORTMNIST

DATASET_ROOTS = {
    "kandinsky_train": "data/kandinsky/train/all_imgs",
    "kandinsky_val": "data/kandinsky/val/all_imgs",
    "kandinsky_test": "data/kandinsky/test/all_imgs",
    "sddoia_train": "data/sddoia/train/",
    "sddoia_val": "data/sddoia/val/",
    "sddoia_test": "data/sddoia/test/",
    "sddoia_ood": "data/sddoia/ood/",
    "shapes3d_train": "../data/shapes3d/images",
    "shapes3d_val": "../data/shapes3d/val/images",
}

LABEL_FILES = {
    "kandinsky": "data/kandinsky_classes.txt",
    "sddoia": "data/sddoia_classes.txt",
    "shapes3d": "data/shapes3d_classes.txt",
    "shortcutmnist": "data/mnist_classes.txt",
}
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class FalseArgs:
    def __init__(self) -> None:
        self.task = "addition"
        self.c_sup = 1
        self.which_c = [-1]
        self.batch_size = 64
        self.stance = 0

    def change_stance(self):
        self.stance = 1


class idx_Dataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
    return preprocess


def get_data(dataset_name, preprocess=None, stance=0):
    if dataset_name in [
        "shortcutmnist_train",
        "shortcutmnist_val",
        "shortcutmnist_test",
        "shortcutmnist_ood",
    ]:
        print("Going to MNIST")

        false_args = FalseArgs()
        if stance == 1:
            false_args.change_stance()

        shortmnist = SHORTMNIST(false_args)

        shortmnist.get_data_loaders()

        if dataset_name == "shortcutmnist_train":
            return shortmnist.dataset_train
        elif dataset_name == "shortcutmnist_val":
            return shortmnist.dataset_val
        elif dataset_name == "shortcutmnist_test":
            return shortmnist.dataset_test
        elif dataset_name == "shortcutmnist_ood":
            return shortmnist.ood_test
        else:
            return NotImplementedError("bah")

    elif dataset_name in DATASET_ROOTS.keys():

        print(DATASET_ROOTS[dataset_name])

        data = idx_Dataset(DATASET_ROOTS[dataset_name], preprocess)
    else:
        NotImplementedError("Wrong")
    return data


def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets


def get_target_model(target_name, device):

    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    return target_model, preprocess
