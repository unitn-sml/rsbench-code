import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np, glob
import re
import matplotlib.pyplot as plt
import json
from torchvision.datasets.folder import pil_loader
from datasets.utils.mnist_creation import generate_r_seq


CONCEPTS_ORDER = {
    "red_light": 3,
    "green_light": 0,
    "car": 5,
    "person": 6,
    "rider": 7,
    "other_obstacle": 8,
    "follow": 1,
    "stop_sign": 4,
    "left_lane": 18,
    "left_green_light": 19,
    "left_follow": 20,
    "no_left_lane": 9,
    "left_obstacle": 10,
    "left_solid_line": 11,
    "right_lane": 12,
    "right_green_light": 13,
    "right_follow": 14,
    "no_right_lane": 15,
    "right_obstacle": 16,
    "right_solid_line": 17,
    "clear": 2,
}

PREFIX = "MINI_BOIA_"


class SDDOIADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        split,
        c_sup=1,
        which_c=[-1],
        return_embeddings=False,
        is_ood_k=False,
    ):

        # path and train/val/test type
        self.base_path = base_path
        self.split = split

        # collecting images
        self.list_images = glob.glob(os.path.join(self.base_path, self.split, "*"))
        # sort the images
        self.list_images = sorted(self.list_images, key=self._extract_number)


        # ok transform
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.labels, self.concepts = [], []

        # sort the keys of the dictionary
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)

        # lmao
        new_images = self.list_images.copy()

        # whether to return the embeddings, means to return also the name of the image
        self.return_embeddings = return_embeddings
        self.names = []

        scenes_name = "scenes"
        if is_ood_k:
            scenes_name = "scenes_ambulance"

        # extract labels and concepts
        for item in self.list_images:
            name = os.path.splitext(os.path.basename(item))[0]
            # extract the ids out of the images
            target_id = name.split("_")[-1]

            # get the target scene
            target_scene = os.path.join(
                self.base_path,
                scenes_name,
                PREFIX + str(target_id) + ".json",
            )

            if not os.path.exists(target_scene):
                new_images.remove(
                    os.path.join(
                        self.base_path, self.split, PREFIX + str(target_id) + ".png"
                    )
                )
                continue

            # concepts and labels
            concepts, labels = [], []

            with open(target_scene, "r") as f:
                # Load the JSON data
                data = json.load(f)

            # take the label
            label = data["label"]
            c = data["concepts"]

            # read out the concepts
            concept_values = [int(c[key]) for key in sorted_concepts]

            labels = np.array(label)
            self.labels.append(labels)

            concepts = np.array(concept_values)
            self.concepts.append(concepts)

            # append the name
            self.names.append(name)

        r_seq = generate_r_seq(len(self.list_images))

        # filter for the concept supervision given
        numel = 0

        if c_sup != 1:
            for i in range(len(self.list_images)):
                if r_seq[i] > c_sup:
                    self.concepts[i][:] = -1
                elif not (which_c[0] == -1):
                    for c in range(self.concepts.shape[1]):
                        if c not in which_c:
                            self.concepts[i, c] = -1
                else:
                    numel += 1
                    for k, order in CONCEPTS_ORDER.items():
                        if k not in [
                            "red_light",
                            "green_light",
                            "car",
                            "person",
                            "rider",
                            "other_obstacle",
                            "stop_sign",
                            "right_green_light",
                            "left_green_light",
                        ]:
                            self.concepts[i][order] = -1

        # print("Giving supervision to", numel)

        # print(self.concepts.shape)
        self.concepts = np.stack(self.concepts, axis=0)
        self.labels = np.stack(self.labels, axis=0)
        self.list_images = np.array(new_images)
        self.names = np.stack(self.names, axis=0)

        if is_ood_k:
            # File path for saving/loading indices
            indices_file = "data/random_indices.npy"

            # Check if the indices file exists
            if os.path.exists(indices_file):
                print("Loading indices...")
                # Load the indices from the file
                random_indices = np.load(indices_file)
            else:
                print("Saving indices...")
                # Get the total number of elements
                # chosen_elements = []
                # stop = []
                # red = []
                # solid_line = []
                # to_insert = []

                # for i in range(self.concepts.shape[0]):
                #     if self.concepts[i, 3] or self.concepts[i, 4] or self.concepts[i, 11] or self.concepts[i, 17]:
                #         to_insert.append(i)

                #         if self.concepts[i, 3]:
                #             red.append(i)
                #         elif self.concepts[i, 4]:
                #             stop.append(i)
                #         else:
                #             solid_line.append(i)
                #     else:
                #         chosen_elements.append(i)

                # print(len(to_insert), len(red), len(stop), len(solid_line))

                # # Select 500 random indices without replacement
                # if len(to_insert) >= 500:
                #     random_indices = to_insert
                # else:
                #     random_indices = np.random.choice(chosen_elements, 500 - len(to_insert), replace=False)

                random_indices = np.random.choice(
                    self.concepts.shape[0], 500, replace=False
                )

                # Save the random indices to a file
                np.save(indices_file, random_indices)

            # Select elements at these random indices
            self.concepts = self.concepts[random_indices]
            self.labels = self.labels[random_indices]
            self.list_images = self.list_images[random_indices]
            self.names = self.names[random_indices]

    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __getitem__(self, item):

        labels = self.labels[item]
        concepts = self.concepts[item]
        img_path = self.list_images[item]
        names = self.names[item]
        image = pil_loader(img_path)

        if self.return_embeddings:
            return self.transform(image), labels, concepts, names

        return self.transform(image), labels, concepts

    def __len__(self):
        return len(self.list_images)


## --------------------------------------------------------------- ##


class ClIP_SDDOIADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        split,
        c_sup=1,
        which_c=[-1],
        return_embeddings=False,
        is_ood_k=False,
    ):

        # path and train/val/test type
        self.dir_path = base_path
        self.base_path = "data/saved_activations/SDDOIA-preprocessed/"
        # self.base_path = os.path.join(base_path, 'sddoia')

        self.split = split

        # collecting images
        self.list_images = glob.glob(
            os.path.join(
                "data/saved_activations/SDDOIA-preprocessed", self.split, "*"
            )
        )
        # sort the images
        self.list_images = sorted(self.list_images, key=self._extract_number)

        self.labels, self.concepts = [], []

        # sort the keys of the dictionary
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)

        # lmao
        new_images = self.list_images.copy()
        scenes_name = "scenes"

        if is_ood_k:
            scenes_name = "scenes_ambulance"

        # whether to return the embeddings, means to return also the name of the image
        self.return_embeddings = return_embeddings
        self.names = []

        # extract labels and concepts
        for item in self.list_images:
            name = os.path.splitext(os.path.basename(item))[0]
            # extract the ids out of the images
            target_id = name.split("_")[-1]

            # get the target scene
            target_scene = os.path.join(
                self.base_path,
                scenes_name,
                PREFIX + str(target_id) + ".json",
            )

            if not os.path.exists(target_scene):
                new_images.remove(
                    os.path.join(
                        self.base_path, self.split, PREFIX + str(target_id) + ".pt"
                    )
                )
                continue

            # concepts and labels
            concepts, labels = [], []

            with open(target_scene, "r") as f:
                # Load the JSON data
                data = json.load(f)

            # take the label
            label = data["label"]
            c = data["concepts"]

            # read out the concepts
            concept_values = [int(c[key]) for key in sorted_concepts]

            labels = np.array(label)
            self.labels.append(labels)

            concepts = np.array(concept_values)
            self.concepts.append(concepts)

            # append the name
            self.names.append(name)

        r_seq = generate_r_seq(len(self.list_images))

        # filter for the concept supervision given
        for i in range(len(self.list_images)):
            if r_seq[i] > c_sup:
                self.concepts[i][:] = -1
            elif not (which_c[0] == -1):
                for c in range(self.concepts.shape[1]):
                    if c not in which_c:
                        self.concepts[i, c] = -1

        # print(self.concepts.shape)
        self.concepts = np.stack(self.concepts, axis=0)
        self.labels = np.stack(self.labels, axis=0)
        self.list_images = np.array(new_images)
        self.names = np.stack(self.names, axis=0)

        if is_ood_k:
            # Get the total number of elements
            total_elements = self.concepts.shape[0]

            # File path for saving/loading indices
            indices_file = "data/random_indices.npy"

            # Check if the indices file exists
            if os.path.exists(indices_file):
                print("Loading indices...")
                # Load the indices from the file
                random_indices = np.load(indices_file)
            else:
                print("Saving indices...")
                # Get the total number of elements
                # chosen_elements = []
                # stop = []
                # red = []
                # solid_line = []
                # to_insert = []

                # for i in range(self.concepts.shape[0]):
                #     if self.concepts[i, 3] or self.concepts[i, 4] or self.concepts[i, 11] or self.concepts[i, 17]:
                #         to_insert.append(i)

                #         if self.concepts[i, 3]:
                #             red.append(i)
                #         elif self.concepts[i, 4]:
                #             stop.append(i)
                #         else:
                #             solid_line.append(i)
                #     else:
                #         chosen_elements.append(i)

                # print(len(to_insert), len(red), len(stop), len(solid_line))

                # Select 500 random indices without replacement
                # if len(to_insert) >= 500:
                #     random_indices = to_insert
                # else:
                #
                random_indices = np.random.choice(
                    self.concepts.shape[0], 500, replace=False
                )

                # Save the random indices to a file
                np.save(indices_file, random_indices)

            # Select elements at these random indices
            self.concepts = self.concepts[random_indices]
            self.labels = self.labels[random_indices]
            self.list_images = self.list_images[random_indices]
            self.names = self.names[random_indices]

        # IMG_PATH = os.path.join(self.dir_path, 'saved_activations', f'sddoia_{self.split}_clip_ViT-B32.pt')
        # image_features = torch.load(IMG_PATH)

        # TXT_PATH = os.path.join(self.dir_path, 'saved_activations', 'sddoia_filtered_ViT-B32.pt')
        # text_features = torch.load(TXT_PATH)

        # image_features /= torch.norm(image_features, dim=1, keepdim=True)
        # text_features  /= torch.norm(text_features, dim=1, keepdim=True)

        # self.imgs = image_features #.to('cuda') @ text_features.T.to('cuda')
        # # self.imgs = self.embs.to('cuda') # @ self.texts.T.to('cuda')

        # self.imgs = self.imgs.to(torch.float64).detach().cpu().numpy()

    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __getitem__(self, item):

        labels = self.labels[item]
        concepts = self.concepts[item]
        img_path = self.list_images[item]
        names = self.names[item]
        # image = self.imgs[item]
        image = torch.load(img_path).to(torch.float64)

        if self.return_embeddings:
            return image, labels, concepts, names

        return image, labels, concepts

    def __len__(self):
        return len(self.list_images)


if __name__ == "__main__":
    print("Hello World")

    train_data = SDDOIADataset("../../data/mini_boia_out", "train")
    val_data = SDDOIADataset("../../data/mini_boia_out", "val")
    test_data = SDDOIADataset("../../data/mini_boia_out", "test")
    ood_data = SDDOIADataset("../../data/mini_boia_out", "ood")

    img, label, concepts = train_data[0]
    print(img.shape, concepts.shape, label.shape)

    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("lmao.png")
    plt.close()
    quit()
