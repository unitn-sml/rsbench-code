from datasets.utils.base_dataset import BaseDataset, KAND_get_loader
from datasets.utils.kand_creation import KAND_Dataset, miniKAND_Dataset
from backbones.kand_encoder import TripleCNNEncoder, TripleMLP
import time
import os
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
import torch


class MiniKandinsky(BaseDataset):
    NAME = "minikandinsky"

    def get_data_loaders(self):
        start = time.time()

        if not hasattr(self, "dataset_train"):
            self.dataset_train = miniKAND_Dataset(
                base_path="data/kand-3k", split="train", finetuning=False
            )

        if self.args.model == "kandcbm":
            self.dataset_train.mask_concepts("red-and-squares-and-circle")

        self.dataset_val = miniKAND_Dataset(
            # base_path="../../data/kand-3k",
            base_path="data/kand-3k",
            split="val",
        )
        self.dataset_test = miniKAND_Dataset(
            # base_path="../../data/kand-3k",
            base_path="data/kand-3k",
            split="test",
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))  # , '\n len ood', len(dataset_ood))

        if not self.args.preprocess:
            train_loader = KAND_get_loader(
                self.dataset_train, self.args.batch_size, val_test=False
            )
            val_loader = KAND_get_loader(self.dataset_val, 500, val_test=True)
            test_loader = KAND_get_loader(self.dataset_test, 500, val_test=True)
        else:
            train_loader = KAND_get_loader(self.dataset_train, 1, val_test=False)
            val_loader = KAND_get_loader(self.dataset_val, 1, val_test=True)
            test_loader = KAND_get_loader(self.dataset_test, 1, val_test=True)

        return train_loader, val_loader, test_loader

    def give_full_supervision(self):
        if not hasattr(self, "dataset_train"):
            self.dataset_train = miniKAND_Dataset(
                base_path="data/kand-3k", split="train", finetuning=False
            )
        self.dataset_train.concepts = self.dataset_train.original_concepts

    def give_supervision_to(self, data_idx, figure_idx, obj_idx):
        if not hasattr(self, "dataset_train"):
            self.dataset_train = miniKAND_Dataset(
                base_path="data/kand-3k", split="train", finetuning=False
            )
            self.dataset_train.concepts = self.dataset_train.original_concepts
        self.dataset_train.mask_concepts_specific(data_idx, figure_idx, obj_idx)

    def get_train_loader_as_val(self):
        return KAND_get_loader(self.dataset_train, self.args.batch_size, val_test=True)

    def get_concept_labels(self):
        return ["square", "circle", "triangle"], ["red", "yellow", "blue"]

    def get_labels(self):
        return [str(i) for i in range(2)]

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_ltn_supervision_loader(self):
        import itertools
        from torchvision import transforms

        # 1 as batch size and not shuffled
        dataloader = KAND_get_loader(self.dataset_train, 1, val_test=True)

        self._create_dir("data/kand-ltn-supervision")
        self._create_dir("data/kand-ltn-supervision/shape")
        self._create_dir("data/kand-ltn-supervision/shape/square")
        self._create_dir("data/kand-ltn-supervision/shape/not_square")
        self._create_dir("data/kand-ltn-supervision/color")
        self._create_dir("data/kand-ltn-supervision/color/red")
        self._create_dir("data/kand-ltn-supervision/color/not_red")

        counter = [0, 0, 0, 0]
        limit = 1000

        for i, data in enumerate(dataloader):
            images, _, concepts = data

            # get the list of images
            full_image = images.squeeze(0)

            image_list = []
            for i in range(0, 252, 28):
                start = i
                end = start + 28
                image_list.append(full_image[:, :, start:end])

            # list of pil images
            to_pil = transforms.ToPILImage()
            pil_image_list = [to_pil(im) for im in image_list]

            # concept vector
            concept_vector = concepts.squeeze(0)

            # reached the limit
            if all(x > limit for x in counter):
                break

            # shapes
            for i, j in itertools.product(range(3), range(3)):
                img_idx = i * 3 + j

                if concept_vector[i, j] == 0:
                    if counter[0] > limit:
                        continue
                    pil_image_list[img_idx].save(
                        f"data/kand-ltn-supervision/shape/square/{counter[0]}.png"
                    )
                    counter[0] += 1
                else:
                    if counter[1] > limit:
                        continue

                    pil_image_list[img_idx].save(
                        f"data/kand-ltn-supervision/shape/not_square/{counter[1]}.png"
                    )
                    counter[1] += 1

            # colors
            for i, j in itertools.product(range(3), range(3, 6)):
                img_idx = i * 3 + (j - 3)

                if concept_vector[i, j] == 0:
                    if counter[2] > limit:
                        continue
                    pil_image_list[img_idx].save(
                        f"data/kand-ltn-supervision/color/red/{counter[2]}.png"
                    )
                    counter[2] += 1
                else:
                    if counter[3] > limit:
                        continue

                    pil_image_list[img_idx].save(
                        f"data/kand-ltn-supervision/color/not_red/{counter[3]}.png"
                    )
                    counter[3] += 1

        print("Done")

    def get_backbone(self, args=None):

        if self.args.backbone == "neural":
            raise NotImplementedError("If you are looking for  NN, check Kandinksy")

        return TripleMLP(latent_dim=6), None

    def get_split(self):
        return 3, ()

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))

    def get_sup(self, n_imgs=10):
        transform = transforms.Compose([transforms.ToTensor()])
        return [
            torch.stack(
                [
                    transform(pil_loader("./data/kand-3k/sup/shape/square/" + img))
                    for img in os.listdir("./data/kand-3k/sup/shape/square")
                    if img.split(".")[1] == "png"
                ]
            )[:n_imgs],
            torch.stack(
                [
                    transform(pil_loader("./data/kand-3k/sup/shape/not_square/" + img))
                    for img in os.listdir("./data/kand-3k/sup/shape/not_square")
                    if img.split(".")[1] == "png"
                ]
            )[:n_imgs],
            torch.stack(
                [
                    transform(pil_loader("./data/kand-3k/sup/color/red/" + img))
                    for img in os.listdir("./data/kand-3k/sup/color/red")
                    if img.split(".")[1] == "png"
                ]
            )[:n_imgs],
            torch.stack(
                [
                    transform(pil_loader("./data/kand-3k/sup/color/not_red/" + img))
                    for img in os.listdir("./data/kand-3k/sup/color/not_red")
                    if img.split(".")[1] == "png"
                ]
            )[:n_imgs],
        ]


if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(
        preprocess=0,
        finetuning=0,
        batch_size=256,
        n_epochs=20,
        validate=1,
        dataset="minikandinsky",
        lr=0.1,
        exp_decay=1,
        warmup_steps=1,
        wandb=None,
        task="kand",
        model="minikanddpl",
    )

    dset = MiniKandinsky(args)
    _ = dset.get_data_loaders()
