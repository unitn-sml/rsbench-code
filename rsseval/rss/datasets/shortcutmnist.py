from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST
from backbones.addmnist_joint import MNISTPairsEncoder, MNISTPairsDecoder
from backbones.addmnist_single import MNISTSingleEncoder
from backbones.mnistcnn import MNISTAdditionCNN
from backbones.disjointmnistcnn import DisjointMNISTAdditionCNN
import numpy as np
from copy import deepcopy
import os


class SHORTMNIST(BaseDataset):
    NAME = "shortmnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):

        if self.args.model == "mnistcbm":
            which_c = [4, 9, 3, 8]

        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup, which_c=self.args.which_c, args=self.args
        )

        ood_test = self.get_ood_test(dataset_test)
        ood_test_2 = self.get_ood_test_2(dataset_test)

        self.filtrate(dataset_train, dataset_val, dataset_test)

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.ood_test = ood_test
        self.ood_test_2 = ood_test_2

        self._compute_class_weights()

        self.train_loader = get_loader(
            dataset_train,
            self.args.batch_size,
            val_test=False,
            sampler=self.train_sampler,
        )
        self.val_loader = get_loader(
            dataset_val, self.args.batch_size, val_test=True, sampler=self.val_sampler
        )
        self.test_loader = get_loader(
            dataset_test, self.args.batch_size, val_test=True, sampler=self.test_sampler
        )
        self.ood_loader = get_loader(
            ood_test, self.args.batch_size, val_test=False, sampler=self.ood_sampler
        )
        self.ood_loader_2 = get_loader(
            ood_test_2, self.args.batch_size, val_test=False, sampler=self.ood_sampler_2
        )

        return self.train_loader, self.val_loader, self.test_loader

    def _compute_class_weights(self):
        import torch
        from torch.utils.data import WeightedRandomSampler

        # Convert data and targets to torch tensors
        train_targets = torch.tensor(self.dataset_train.targets, dtype=torch.int64)
        val_targets = torch.tensor(self.dataset_val.targets, dtype=torch.int64)
        test_targets = torch.tensor(self.dataset_test.targets, dtype=torch.int64)
        ood_targets = torch.tensor(self.ood_test.targets, dtype=torch.int64)
        ood_targets_2 = torch.tensor(self.ood_test_2.targets, dtype=torch.int64)

        # Calculate class weights
        train_class_weights = 1.0 / torch.bincount(train_targets).float()
        val_class_weights = 1.0 / torch.bincount(val_targets).float()
        test_class_weights = 1.0 / torch.bincount(test_targets).float()
        ood_class_weights = 1.0 / torch.bincount(ood_targets).float()
        ood_class_weights_2 = 1.0 / torch.bincount(ood_targets_2).float()

        # Assign a weight to each sample
        train_sample_weights = train_class_weights[train_targets]
        val_sample_weights = val_class_weights[val_targets]
        test_sample_weights = test_class_weights[test_targets]
        ood_sample_weights = ood_class_weights[ood_targets]
        ood_sample_weights_2 = ood_class_weights_2[ood_targets_2]

        # Create WeightedRandomSampler
        self.train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True,
        )
        self.val_sampler = WeightedRandomSampler(
            weights=val_sample_weights,
            num_samples=len(val_sample_weights),
            replacement=True,
        )
        self.test_sampler = WeightedRandomSampler(
            weights=test_sample_weights,
            num_samples=len(test_sample_weights),
            replacement=True,
        )
        self.ood_sampler = WeightedRandomSampler(
            weights=ood_sample_weights,
            num_samples=len(ood_sample_weights),
            replacement=True,
        )
        self.ood_sampler_2 = WeightedRandomSampler(
            weights=ood_sample_weights_2,
            num_samples=len(ood_sample_weights_2),
            replacement=True,
        )

    def get_backbone(self):
        if self.args.joint:
            if self.args.backbone == "neural":
                return MNISTAdditionCNN(), None
            return MNISTPairsEncoder(), MNISTPairsDecoder()
        else:

            if self.args.backbone == "neural":
                return DisjointMNISTAdditionCNN(n_images=self.get_split()[0]), None

            return MNISTSingleEncoder(), MNISTPairsDecoder()

    def get_split(self):
        if self.args.joint:
            return 1, (10, 10)
        else:
            return 2, (10,)

    def get_concept_labels(self):
        return [str(i) for i in range(10)]

    def get_labels(self):
        return [str(i) for i in range(9)]

    def give_supervision_to(self, percentage):
        pass  # TODO

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

        train_dataset.data = train_dataset.data[train_mask]
        val_dataset.data = val_dataset.data[val_mask]
        test_dataset.data = test_dataset.data[test_mask]

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

        ood_test.data = ood_test.data[test_mask]
        ood_test.concepts = ood_test.concepts[test_mask]
        print(
            len(test_dataset),
            len(ood_test),
            len(ood_test.targets),
            ood_test.targets.shape,
        )
        ood_test.targets = np.array(ood_test.targets)[test_mask]

        return ood_test

    def get_ood_test_2(self, test_dataset):

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
        test_mask_concepts = np.logical_and(test_mask_in_range, test_mask_value)
        test_mask_sum = (
            (test_dataset.targets[:] == 6)
            | (test_dataset.targets[:] == 10)
            | (test_dataset.targets[:] == 12)
        )

        test_mask = np.logical_and(test_mask_concepts, test_mask_sum)

        ood_test.data = ood_test.data[test_mask]
        ood_test.concepts = ood_test.concepts[test_mask]
        print(
            len(test_dataset),
            len(ood_test),
            len(ood_test.targets),
            ood_test.targets.shape,
        )
        ood_test.targets = np.array(ood_test.targets)[test_mask]

        return ood_test

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test samples", len(self.dataset_test.data))
        print("Test OOD samples", len(self.ood_test.data))

    def save_mnist_tcav_loader_2digits(
        self, d_type: str, folder_name="shortcutmnist-tcav-2-digits"
    ):
        from torchvision import transforms

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
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(full_tensor)

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
                pil_image.save(
                    f"data/{folder_name}/{concept_1}_x/{counter[concept_1]}.png"
                )
                counter[concept_1] += 1

            if not counter[concept_2 + 10] > limit:
                pil_image.save(
                    f"data/{folder_name}/x_{concept_2}/{counter[concept_2 + 10]}.png"
                )
                counter[concept_2 + 10] += 1

        print("Total", counter)

        print("Done")

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":
    from argparse import Namespace

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

    dataset = SHORTMNIST(args)

    train, val, test = dataset.get_data_loaders()
    dataset.save_mnist_tcav_loader_2digits("val")
