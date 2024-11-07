from datasets.utils.sddoia_creation import ClIP_SDDOIADataset
from backbones.identity import Identity
from backbones.disent_encoder_decoder import DecoderConv64
import time

from datasets.utils.base_dataset import BaseDataset, SDDOIA_get_loader
from datasets.utils.sddoia_creation import CONCEPTS_ORDER
from datasets.utils.presddoia_creation import PreSDDOIADataset
from backbones.sddoiacnn import FFNN
from backbones.presddoiacnn import PreSDDOIAMlp

import torch
import os
import numpy as np
from argparse import Namespace


class CLIPSDDOIA(BaseDataset):
    NAME = "clipsddoia"

    def __init__(self, args) -> None:
        super().__init__(args)

    def get_backbone(self, args=None):
        return Identity(21, 1), None

    def get_split(self):
        return 3, ()

    def print_stats(self):
        print("Hello CLIP")

    def get_data_loaders(self):
        start = time.time()

        self.dataset_train = ClIP_SDDOIADataset(
            base_path="data",
            split="train",
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        self.dataset_val = ClIP_SDDOIADataset(base_path="data", split="val")
        self.dataset_test = ClIP_SDDOIADataset(base_path="data", split="test")
        self.dataset_ood = ClIP_SDDOIADataset(base_path="data", split="ood")
        self.dataset_ood_ambulance = ClIP_SDDOIADataset(
            base_path="data", split="ood_ambulance", is_ood_k=True
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        self.train_loader = SDDOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = SDDOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = SDDOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = SDDOIA_get_loader(
            self.dataset_ood, self.args.batch_size, val_test=True
        )
        self.ood_loader_ambulance = SDDOIA_get_loader(
            self.dataset_ood_ambulance, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        # if self.args.backbone == "neural":
        #     return
        #     return Identity(21,1), None
        return FFNN().to(torch.float64), None

    def get_split(self):
        return 1, ()

    def get_concept_labels(self):
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)
        return sorted_concepts

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
        print("Test OOD samples", len(self.dataset_ood))

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_minibboia_tcav_loader(self, d_type: str, folder_name="sddoia-tcav-clip"):

        # 1 as batch size and not shuffled
        if d_type == "train":
            dataloader = SDDOIA_get_loader(self.dataset_train, 1, val_test=True)
        elif d_type == "val":
            dataloader = SDDOIA_get_loader(self.dataset_val, 1, val_test=True)
        else:
            dataloader = SDDOIA_get_loader(self.dataset_test, 1, val_test=True)

        self._create_dir(f"data/{folder_name}")

        counter = []
        for c in CONCEPTS_ORDER.keys():
            self._create_dir(f"data/{folder_name}/{c}")
            counter.append(0)

        # count = 0
        np.random.seed(42)
        limit = 1000

        for i, data in enumerate(dataloader):
            images, _, concepts = data

            # get the list of images
            full_tensor = images.squeeze(0)

            # concept vector
            concept_vector = concepts.squeeze(0)

            # reached the limit
            if all(x > limit for x in counter):
                break

            index_concept_dict = {value: key for key, value in CONCEPTS_ORDER.items()}

            concept_vector_indices = np.arange(len(concept_vector))
            np.random.shuffle(concept_vector_indices)

            # select concept
            for c in concept_vector_indices:
                concept_name = index_concept_dict[c]
                # already encountered the limit
                if counter[c] > limit or concept_vector[c] == 0:
                    continue

                torch.save(
                    full_tensor, f"data/{folder_name}/{concept_name}/{counter[c]}.pt"
                )
                counter[c] += 1

                break

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
        dataset="boia",
        lr=0.001,
        exp_decay=0.99,
        warmup_steps=1,
        wandb=None,
        task="boia",
        boia_model="ce",
        model="sddoiadpl",
        c_sup=0,
        which_c=-1,
    )

    dataset = CLIPSDDOIA(args)

    train, val, test = dataset.get_data_loaders()
    dataset.save_minibboia_tcav_loader("val")
