from datasets.utils.base_dataset import BaseDataset, KAND_get_loader
from datasets.utils.kand_creation import PreKAND_Dataset
from backbones.simple_encoder import SimpleMLP
from backbones.disent_encoder_decoder import DecoderConv64
import time


class PreKandinsky(BaseDataset):
    NAME = "prekandinsky"

    def get_data_loaders(self):
        start = time.time()

        dataset_train = PreKAND_Dataset(base_path="data/kand-preprocess", split="train")
        dataset_val = PreKAND_Dataset(base_path="data/kand-preprocess", split="val")
        dataset_test = PreKAND_Dataset(base_path="data/kand-preprocess", split="test")
        # dataset_ood   = KAND_Dataset(base_path='data/kandinsky/data',split='ood')

        dataset_train.mask_concepts("red-and-squares")

        print(f"Loaded datasets in {time.time()-start} s.")

        print("Len loaders: \n train:", len(dataset_train), "\n val:", len(dataset_val))
        print(" len test:", len(dataset_test))  # , '\n len ood', len(dataset_ood))

        train_loader = KAND_get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        val_loader = KAND_get_loader(dataset_val, 1000, val_test=True)
        test_loader = KAND_get_loader(dataset_test, 1000, val_test=True)

        # self.ood_loader = get_loader(dataset_ood,  self.args.batch_size, val_test=True)

        return train_loader, val_loader, test_loader

    def get_backbone(self, args=None):
        return SimpleMLP(z_dim=18, z_multiplier=2), DecoderConv64(
            x_shape=(3, 64, 64), z_size=18, z_multiplier=2
        )

    def get_split(self):
        return 3, ()
