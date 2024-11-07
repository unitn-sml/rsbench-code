from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, SDDOIA_get_loader
from datasets.utils.sddoia_creation import SDDOIADataset, CONCEPTS_ORDER
from backbones.resnet import ResNetEncoder
from backbones.sddoiacnn import SDDOIACnn
import time


class SDDOIA(BaseDataset):
    NAME = "sddoia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.return_embeddings = False

    def get_data_loaders(self):
        start = time.time()

        self.dataset_train = SDDOIADataset(
            base_path="data/mini_boia_out",
            split="train",
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
            return_embeddings=self.return_embeddings,
        )
        self.dataset_val = SDDOIADataset(
            base_path="data/mini_boia_out",
            split="val",
            return_embeddings=self.return_embeddings,
        )
        self.dataset_test = SDDOIADataset(
            base_path="data/mini_boia_out",
            split="test",
            return_embeddings=self.return_embeddings,
        )
        self.dataset_ood = SDDOIADataset(
            base_path="data/mini_boia_out",
            split="ood",
            return_embeddings=self.return_embeddings,
        )
        self.dataset_ood_ambulance = SDDOIADataset(
            base_path="data/mini_boia_out",
            split="ood_ambulance",
            return_embeddings=self.return_embeddings,
            is_ood_k=True,
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        keep_order = True if self.return_embeddings else False
        self.train_loader = SDDOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=keep_order
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
        if self.args.backbone == "neural":
            return SDDOIACnn(), None

        if not self.return_embeddings:
            return ResNetEncoder(c_dim=21), None
        else:
            return ResNetEncoder(c_dim=None), None

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
