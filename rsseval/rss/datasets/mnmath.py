from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, MNMATH_get_loader
from datasets.utils.mnmath_creation import MNMATHDataset
from backbones.cnnnosharing import CBMNoSharing, MNMNISTCNN
from backbones.addmnist_single import MNISTSingleEncoder
import time


class MNMATH(BaseDataset):
    NAME = "mnmath"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.return_embeddings = False

    def get_data_loaders(self):
        start = time.time()

        self.dataset_train = MNMATHDataset(
            base_path="data/mnmath",
            split="train",
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )

        self.dataset_val = MNMATHDataset(
            base_path="data/mnmath",
            split="val",
        )
        self.dataset_test = MNMATHDataset(
            base_path="data/mnmath",
            split="test",
        )
        self.dataset_ood = MNMATHDataset(
            base_path="data/mnmath",
            split="ood",
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
        self.train_loader = MNMATH_get_loader(
            self.dataset_train, self.args.batch_size, val_test=keep_order
        )
        self.val_loader = MNMATH_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = MNMATH_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = MNMATH_get_loader(
            self.dataset_ood, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return MNMNISTCNN(num_images=8, num_classes=2), None

        if self.args.task == "mnmath":
            return MNISTSingleEncoder(), None # IndividualMNISTCNN(10), None
        return CBMNoSharing(num_images=8, nout=10), None

    def get_split(self):
        return 4, ()

    def get_concept_labels(self):
        return [0, 1]

    def get_labels(self):
        return [0, 1]

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
        print("Test OOD samples", len(self.dataset_ood))

if __name__ == "__main__":
    dataset = MNMATH()

    for batch_idx, data in enumerate(dataset.train_loader):
        images, labels, concepts = data

        print(images[0].shape)
        print(labels[0])
        print(concepts[0])
        quit()