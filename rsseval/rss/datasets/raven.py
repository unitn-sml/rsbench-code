
from datasets.utils.base_dataset import BaseDataset
from datasets.utils.raven_creation import RAVEN_Dataset
from backbones.raven_encoder import RavenMLP
import time
from torch.utils.data import DataLoader

class RAVEN(BaseDataset):
    NAME = "raven"

    def get_data_loaders(self):
        start = time.time()
        
        # We can pass specific args for configuration if needed via self.args
        # For now defaults to center_single
        
        config = getattr(self.args, 'raven_config', "center_single")

        self.dataset_train = RAVEN_Dataset(
            base_path="data/RAVEN-10000",
            config=config,
            split="train",
        )
        
        self.dataset_val = RAVEN_Dataset(
            base_path="data/RAVEN-10000",
            config=config,
            split="val",
        )
        
        self.dataset_test = RAVEN_Dataset(
            base_path="data/RAVEN-10000",
            config=config,
            split="test",
        )

        print(f"Loaded datasets in {time.time()-start:.2f} s.")
        self.print_stats()

        # Create DataLoaders
        # Note: BaseDataset usually expects get_data_loaders to return loaders, not just set datasets
        
        train_loader = DataLoader(self.dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.dataset_val, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(self.dataset_test, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def get_backbone(self, args=None):
        # Return the encoder compatible with 160x160 input
        # We return (encoder, decoder). Decoder is None for now.
        return RavenMLP(), None

    def get_split(self):
        # 16 images per sample
        return 16, ()

    def get_concept_labels(self):
        # Return labels for visualization/confusion matrices
        # We have 4 concepts: Type, Size, Color, Number
        return ["Type", "Size", "Color", "Number"], [
            ["Triangle", "Square", "Pentagon", "Hexagon", "Circle"], # Type (0-4)
            ["0", "1", "2", "3", "4", "5"], # Size (0-5)
            [str(i) for i in range(10)], # Color (0-9)
            [str(i) for i in range(9)] # Number (0-8)
        ]

    def get_labels(self):
        # 8 choices (0-7)
        return [str(i) for i in range(8)]

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))

if __name__ == "__main__":
    from argparse import Namespace
    dataset = RAVEN(
        args=Namespace(batch_size=32)
    )
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    print(f"Training number of batches: {len(train_loader)}")
    print(f"Validation number of batches: {len(val_loader)}")
    print(f"Test number of batches: {len(test_loader)}")