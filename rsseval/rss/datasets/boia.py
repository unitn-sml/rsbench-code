from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, BOIA_get_loader
from datasets.utils.boia_creation import BOIADataset
from datasets.utils.miniboia_creation import CONCEPTS_ORDER
from backbones.boia_linear import BOIAConceptizer
from backbones.boia_mlp import BOIAMLP
import time


class BOIA(BaseDataset):
    NAME = "boia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def get_data_loaders(self):
        start = time.time()

        image_dir = "data/bdd2048/"
        train_data_path = "data/bdd2048/train_BDD_OIA.pkl"
        val_data_path = "data/bdd2048/val_BDD_OIA.pkl"
        test_data_path = "data/bdd2048/test_BDD_OIA.pkl"

        self.dataset_train = BOIADataset(
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
        self.dataset_val = BOIADataset(
            pkl_file_path=val_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "val",
            n_class_attr=2,
            transform=None,
        )
        self.dataset_test = BOIADataset(
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
            return BOIAMLP(), None

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
