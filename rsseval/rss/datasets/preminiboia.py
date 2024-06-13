from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, MiniBOIA_get_loader
from datasets.utils.miniboia_creation import CONCEPTS_ORDER
from datasets.utils.preminiboia_creation import PreMiniBOIADataset
from backbones.miniboia_mlp import MiniBOIALinear
from backbones.preminiboiacnn import PreMiniBOIAMlp
import time


class PreMINIBOIA(BaseDataset):
    NAME = "preminiboia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def get_data_loaders(self):
        start = time.time()

        self.dataset_train = PreMiniBOIADataset(
            base_path="data/mini_boia_embeddings",
            split="train",
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        self.dataset_val = PreMiniBOIADataset(
            base_path="data/mini_boia_embeddings", split="val"
        )
        self.dataset_test = PreMiniBOIADataset(
            base_path="data/mini_boia_embeddings", split="test"
        )
        self.dataset_ood = PreMiniBOIADataset(
            base_path="data/mini_boia_embeddings", split="ood"
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        self.train_loader = MiniBOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = MiniBOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = MiniBOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = MiniBOIA_get_loader(
            self.dataset_ood, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return PreMiniBOIAMlp(), None
        return MiniBOIALinear(din=512, nconcept=21), None

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
