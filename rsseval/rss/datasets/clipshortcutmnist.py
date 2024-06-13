from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.clip_mnst_creation import load_2MNIST
from backbones.identity import Identity
from backbones.addmnist_joint import MNISTPairsEncoder, MNISTPairsDecoder
from backbones.addmnist_single import MNISTSingleEncoder
from backbones.mnistcnn import MNISTAdditionCNN
from backbones.disjointmnistcnn import MLP

import numpy as np
from copy import deepcopy
from argparse import Namespace
import torch
import os


class CLIPSHORTMNIST(BaseDataset):
    NAME = "clipshortmnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup, which_c=self.args.which_c, args=self.args
        )

        ood_test = self.get_ood_test(dataset_test)

        self.filtrate(dataset_train, dataset_val, dataset_test)

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.ood_test = ood_test

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(dataset_val, self.args.batch_size, val_test=True)
        self.test_loader = get_loader(dataset_test, self.args.batch_size, val_test=True)
        self.ood_loader = get_loader(ood_test, self.args.batch_size, val_test=False)

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.joint:
            if self.args.backbone == "neural":
                return MLP(n_images=self.get_split()[0]), None
            return MNISTPairsEncoder(), MNISTPairsDecoder()
        else:
            if self.args.backbone == "neural":
                return MLP(n_images=self.get_split()[0]), None

            return MNISTSingleEncoder(), MNISTPairsDecoder()

    def get_split(self):
        if self.args.joint:
            return 1, (10, 10)
        else:
            return 2, (10,)

    def get_concept_labels(self):
        return [str(i) for i in range(10)]

    def filtrate(self, train_dataset, val_dataset, test_dataset):

        train_c_mask1 = (
            (
                (train_dataset.real_concepts[:, 0] == 0)
                & (train_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 2)
                & (train_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 4)
                & (train_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 4)
                & (train_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 1)
                & (train_dataset.real_concepts[:, 1] == 5)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 3)
                & (train_dataset.real_concepts[:, 1] == 7)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 1)
                & (train_dataset.real_concepts[:, 1] == 9)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 3)
                & (train_dataset.real_concepts[:, 1] == 9)
            )
        )  # | \
        # ((train_dataset.real_concepts[:,0] == 5) & (train_dataset.real_concepts[:,1] == 7))
        train_c_mask2 = (
            (
                (train_dataset.real_concepts[:, 1] == 0)
                & (train_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 2)
                & (train_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 4)
                & (train_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 4)
                & (train_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 1)
                & (train_dataset.real_concepts[:, 0] == 5)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 3)
                & (train_dataset.real_concepts[:, 0] == 7)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 1)
                & (train_dataset.real_concepts[:, 0] == 9)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 3)
                & (train_dataset.real_concepts[:, 0] == 9)
            )
        )  # | \
        # ((train_dataset.real_concepts[:,0] == 7) & (train_dataset.real_concepts[:,1] == 5))
        train_mask = np.logical_or(train_c_mask1, train_c_mask2)

        val_c_mask1 = (
            (
                (val_dataset.real_concepts[:, 0] == 0)
                & (val_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 2)
                & (val_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 4)
                & (val_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 4)
                & (val_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 1)
                & (val_dataset.real_concepts[:, 1] == 5)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 3)
                & (val_dataset.real_concepts[:, 1] == 7)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 1)
                & (val_dataset.real_concepts[:, 1] == 9)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 3)
                & (val_dataset.real_concepts[:, 1] == 9)
            )
        )  # | \
        #   ((val_dataset.real_concepts[:,0] == 5) & (val_dataset.real_concepts[:,1] == 7))
        val_c_mask2 = (
            (
                (val_dataset.real_concepts[:, 1] == 0)
                & (val_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 2)
                & (val_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 4)
                & (val_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 4)
                & (val_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 1)
                & (val_dataset.real_concepts[:, 0] == 5)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 3)
                & (val_dataset.real_concepts[:, 0] == 7)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 1)
                & (val_dataset.real_concepts[:, 0] == 9)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 3)
                & (val_dataset.real_concepts[:, 0] == 9)
            )
        )  # | \
        #   ((val_dataset.real_concepts[:,1] == 5) & (val_dataset.real_concepts[:,0] == 7))
        val_mask = np.logical_or(val_c_mask1, val_c_mask2)

        test_c_mask1 = (
            (
                (test_dataset.real_concepts[:, 0] == 0)
                & (test_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 2)
                & (test_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 4)
                & (test_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 4)
                & (test_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 1)
                & (test_dataset.real_concepts[:, 1] == 5)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 3)
                & (test_dataset.real_concepts[:, 1] == 7)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 1)
                & (test_dataset.real_concepts[:, 1] == 9)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 3)
                & (test_dataset.real_concepts[:, 1] == 9)
            )
        )  # | \
        #    ((test_dataset.real_concepts[:,0] == 5) & (test_dataset.real_concepts[:,1] == 7))

        test_c_mask2 = (
            (
                (test_dataset.real_concepts[:, 1] == 0)
                & (test_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 2)
                & (test_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 4)
                & (test_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 4)
                & (test_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 1)
                & (test_dataset.real_concepts[:, 0] == 5)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 3)
                & (test_dataset.real_concepts[:, 0] == 7)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 1)
                & (test_dataset.real_concepts[:, 0] == 9)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 3)
                & (test_dataset.real_concepts[:, 0] == 9)
            )
        )  # | \
        #    ((test_dataset.real_concepts[:,1] == 5) & (test_dataset.real_concepts[:,0] == 7))

        test_mask = np.logical_or(test_c_mask1, test_c_mask2)

        # train_dataset.data = train_dataset.data[train_mask]
        # val_dataset.data = val_dataset.data[val_mask]
        # test_dataset.data = test_dataset.data[test_mask]

        train_dataset.concepts = train_dataset.concepts[train_mask]
        val_dataset.concepts = val_dataset.concepts[val_mask]
        test_dataset.concepts = test_dataset.concepts[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        val_dataset.targets = np.array(val_dataset.targets)[val_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    def get_ood_test(self, test_dataset):

        ood_test = deepcopy(test_dataset)

        mask_col0 = (test_dataset.real_concepts[:, 0] >= 0) & (
            test_dataset.real_concepts[:, 1] <= 9
        )
        mask_col1 = (test_dataset.real_concepts[:, 1] >= 0) & (
            test_dataset.real_concepts[:, 0] <= 9
        )

        test_c_mask1 = (
            (
                (test_dataset.real_concepts[:, 0] == 0)
                & (test_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 2)
                & (test_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 4)
                & (test_dataset.real_concepts[:, 1] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 4)
                & (test_dataset.real_concepts[:, 1] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 1)
                & (test_dataset.real_concepts[:, 1] == 5)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 3)
                & (test_dataset.real_concepts[:, 1] == 7)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 1)
                & (test_dataset.real_concepts[:, 1] == 9)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 3)
                & (test_dataset.real_concepts[:, 1] == 9)
            )
        )

        test_c_mask2 = (
            (
                (test_dataset.real_concepts[:, 1] == 0)
                & (test_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 2)
                & (test_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 4)
                & (test_dataset.real_concepts[:, 0] == 6)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 4)
                & (test_dataset.real_concepts[:, 0] == 8)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 1)
                & (test_dataset.real_concepts[:, 0] == 5)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 3)
                & (test_dataset.real_concepts[:, 0] == 7)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 1)
                & (test_dataset.real_concepts[:, 0] == 9)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 3)
                & (test_dataset.real_concepts[:, 0] == 9)
            )
        )

        test_mask_in_range = np.logical_and(mask_col0, mask_col1)
        test_mask_value = np.logical_and(~test_c_mask1, ~test_c_mask2)

        test_mask = np.logical_and(test_mask_in_range, test_mask_value)

        # ood_test.data = ood_test.data[test_mask]
        ood_test.concepts = ood_test.concepts[test_mask]
        ood_test.targets = np.array(ood_test.targets)[test_mask]
        return ood_test

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test samples", len(self.dataset_test.data))
        print("Test OOD samples", len(self.ood_test.data))

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_mnist_tcav_loader(
        self, d_type: str, folder_name="clip-shortcutmnist-tcav"
    ):

        # 1 as batch size and not shuffled
        if d_type == "train":
            dataloader = get_loader(self.dataset_train, 1, val_test=True)
        elif d_type == "val":
            dataloader = get_loader(self.dataset_val, 1, val_test=True)
        else:
            dataloader = get_loader(self.dataset_test, 1, val_test=True)

        self._create_dir(f"data/{folder_name}")

        counter = []
        for c in range(10):
            counter.append(0)

        # count = 0
        np.random.seed(42)
        limit = 1000

        for i, data in enumerate(dataloader):
            images, _, concepts = data

            # get the list of images
            full_tensor = images.squeeze(0)

            # concept vector
            concept = concepts.squeeze(0)[1].item()

            # reached the limit
            if all(x > limit for x in counter):
                break

            # create folder if needed
            if counter[concept] == 0:
                self._create_dir(f"data/{folder_name}/{concept}")

            # select concept
            if counter[concept] > limit:
                continue

            torch.save(
                full_tensor, f"data/{folder_name}/{concept}/{counter[concept]}.pt"
            )
            counter[concept] += 1

        print("Total", counter)

        print("Done")

    def save_mnist_tcav_loader_2digits(
        self, d_type: str, folder_name="clip-shortcutmnist-tcav-2"
    ):

        # 1 as batch size and not shuffled
        if d_type == "train":
            dataloader = get_loader(self.dataset_train, 1, val_test=True)
        elif d_type == "val":
            dataloader = get_loader(self.dataset_val, 1, val_test=True)
        else:
            dataloader = get_loader(self.dataset_test, 1, val_test=True)

        self._create_dir(f"data/{folder_name}")

        counter = []
        for c in range(20):
            counter.append(0)

        # count = 0
        np.random.seed(42)
        limit = 1000

        for i, data in enumerate(dataloader):
            images, _, concepts = data

            # get the list of images
            full_tensor = images.squeeze(0)

            # concept vectors
            concept_1 = concepts.squeeze(0)[0].item()
            concept_2 = concepts.squeeze(0)[1].item()

            # reached the limit
            if all(x > limit for x in counter):
                break

            # create folder if needed
            if counter[concept_1] == 0:
                self._create_dir(f"data/{folder_name}/{concept_1}_x")

            if counter[concept_2 + 10] == 0:
                self._create_dir(f"data/{folder_name}/x_{concept_2}")

            # select concept
            if counter[concept_1] > limit and counter[concept_2 + 10] > limit:
                continue

            if not counter[concept_1] > limit:
                torch.save(
                    full_tensor,
                    f"data/{folder_name}/{concept_1}_x/{counter[concept_1]}.pt",
                )
                counter[concept_1] += 1

            if not counter[concept_2 + 10] > limit:
                torch.save(
                    full_tensor,
                    f"data/{folder_name}/x_{concept_2}/{counter[concept_2 + 10]}.pt",
                )
                counter[concept_2 + 10] += 1

        print("Total", counter)

        print("Done")


if __name__ == "__main__":
    args = Namespace(
        backbone="neural",
        preprocess=0,
        finetuning=0,
        batch_size=256,
        n_epochs=20,
        validate=1,
        dataset="shortminst",
        lr=0.001,
        exp_decay=0.99,
        warmup_steps=1,
        wandb=None,
        task="addition",
        boia_model="ce",
        model="mnistdpl",
        c_sup=0,
        which_c=-1,
    )

    dataset = CLIPSHORTMNIST(args)

    train, val, test = dataset.get_data_loaders()
    dataset.save_mnist_tcav_loader_2digits("val")
