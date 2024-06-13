from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST
from backbones.addmnist_joint import MNISTPairsEncoder, MNISTPairsDecoder
from backbones.addmnist_repeated import MNISTRepeatedEncoder
from backbones.addmnist_single import MNISTSingleEncoder
from backbones.mnistcnn import MNISTAdditionCNN
from backbones.disjointmnistcnn import DisjointMNISTAdditionCNN


class ADDMNIST(BaseDataset):
    NAME = "addmnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup, which_c=self.args.which_c, args=self.args
        )

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(dataset_val, self.args.batch_size, val_test=True)
        self.test_loader = get_loader(dataset_test, self.args.batch_size, val_test=True)

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.joint:
            if not self.args.splitted:
                return MNISTPairsEncoder(), MNISTPairsDecoder()
            else:
                if self.args.backbone == "neural":
                    return MNISTAdditionCNN(), None
                return MNISTRepeatedEncoder(), MNISTPairsDecoder()
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
        return [str(i) for i in range(19)]

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test samples", len(self.dataset_test.data))
