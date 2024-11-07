from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, BOIA_get_loader
from datasets.utils.boia_creation import CLIPBOIADataset
from datasets.utils.sddoia_creation import CONCEPTS_ORDER
from backbones.boia_linear import BOIAConceptizer
from backbones.boia_mlp import CLIPMLP
import time
import os
import numpy as np
import torch


class CLIPBOIA(BaseDataset):
    NAME = "clipboia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def get_data_loaders(self):
        start = time.time()

        image_dir = "data/bdd2048/"

        train_data_path = "/data/bdd2048/train_BDD_OIA.pkl"
        val_data_path = "data/bdd2048/val_BDD_OIA.pkl"
        test_data_path = "data/bdd2048/test_BDD_OIA.pkl"

        self.dataset_train = CLIPBOIADataset(
            pkl_file_path=train_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "train",
            n_class_attr=2,
            transform=None,
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        self.dataset_val = CLIPBOIADataset(
            pkl_file_path=val_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "val",
            n_class_attr=2,
            transform=None,
        )
        self.dataset_test = CLIPBOIADataset(
            pkl_file_path=test_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "test",
            n_class_attr=2,
            transform=None,
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        self.train_loader = BOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = BOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = BOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return CLIPMLP(), None

        return BOIAConceptizer(din=2048, nconcept=21), None

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

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_minibboia_tcav_loader(self, d_type: str, folder_name="boia-tcav-clip"):

        # 1 as batch size and not shuffled
        if d_type == "train":
            dataloader = BOIA_get_loader(self.dataset_train, 1, val_test=True)
        elif d_type == "val":
            dataloader = BOIA_get_loader(self.dataset_val, 1, val_test=True)
        else:
            dataloader = BOIA_get_loader(self.dataset_test, 1, val_test=True)

        self._create_dir(f"data/{folder_name}")

        counter = []
        for c in CONCEPTS_ORDER.keys():
            self._create_dir(f"data/{folder_name}/{c}")
            counter.append(0)

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

    dataset = CLIPBOIA(args)

    train, val, test = dataset.get_data_loaders()
    dataset.save_minibboia_tcav_loader("val")
