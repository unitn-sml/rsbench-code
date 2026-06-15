
from datasets.utils.base_dataset import BaseDataset
from datasets.utils.raven_creation import RAVEN_Dataset
from backbones.raven_encoder import RavenMLP
import time
from torch.utils.data import DataLoader

# ── Concept labels per n_values ──────────────────────────────────────
_TYPE_LABELS = {
    3: ["Triangle", "Pentagon", "Circle"],
    4: ["Triangle", "Square", "Pentagon", "Circle"],
}
_SIZE_LABELS = {
    3: ["Small", "Medium", "Large"],
    4: ["Small", "Medium-small", "Medium-large", "Large"],
}
_COLOR_LABELS = {
    3: ["White", "Gray", "Black"],
    4: ["White", "Light-gray", "Dark-gray", "Black"],
}

class RAVEN(BaseDataset):
    NAME = "raven"

    def get_data_loaders(self):
        start = time.time()

        config = getattr(self.args, 'raven_config', "center_single")
        n = getattr(self.args, 'n_values', 3)
        base_path = f"data/RAVEN-{n}x{n}x{n}"

        self.dataset_train = RAVEN_Dataset(
            base_path=base_path,
            config=config,
            split="train",
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        
        self.dataset_val = RAVEN_Dataset(
            base_path=base_path,
            config=config,
            split="val",
        )
        
        self.dataset_test = RAVEN_Dataset(
            base_path=base_path,
            config=config,
            split="test",
        )

        print(f"Loaded datasets in {time.time()-start:.2f} s.")
        self.print_stats()
        
        train_loader = DataLoader(self.dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.dataset_val, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(self.dataset_test, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def get_backbone(self, args=None):
        n = getattr(self.args, 'n_values', 3) if self.args else 3
        return RavenMLP(latent_dim=n * 3), None

    def get_split(self):
        return 16, ()

    def get_concept_labels(self):
        n = getattr(self.args, 'n_values', 3)
        return ["Type", "Size", "Color", "Number"], [
            _TYPE_LABELS.get(n, _TYPE_LABELS[3]),
            _SIZE_LABELS.get(n, _SIZE_LABELS[3]),
            _COLOR_LABELS.get(n, _COLOR_LABELS[3]),
            [str(i) for i in range(9)],
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