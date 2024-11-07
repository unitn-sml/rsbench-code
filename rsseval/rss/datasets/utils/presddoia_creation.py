import os
import torch
import torch.utils.data
import numpy as np, glob
import re
import json
from datasets.utils.mnist_creation import generate_r_seq
from datasets.utils.sddoia_creation import CONCEPTS_ORDER

PREFIX = "MINI_BOIA_"


class PreSDDOIADataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, c_sup=1, which_c=[-1]):

        # path and train/val/test type
        self.base_path = base_path
        self.split = split

        # collecting embeddings
        self.list_embeddings = glob.glob(os.path.join(self.base_path, self.split, "*"))
        # sort the embeddings
        self.list_embeddings = sorted(self.list_embeddings, key=self._extract_number)

        # list of labels and concepts
        self.labels, self.concepts = [], []

        # sort the keys of the dictionary
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)

        # new copy of the embeddings
        new_list_embeddings = self.list_embeddings.copy()

        # extract labels and concepts
        for item in self.list_embeddings:
            name = os.path.splitext(os.path.basename(item))[0]
            # extract the ids out of the images
            target_id = name.split("_")[-1]

            # get the target scene
            target_scene = os.path.join(
                self.base_path,
                "scenes",
                PREFIX + str(target_id) + ".json",
            )

            if not os.path.exists(target_scene):
                new_list_embeddings.remove(
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

        r_seq = generate_r_seq(len(self.list_embeddings))

        # filter for the concept supervision given
        for i in range(len(self.list_embeddings)):
            if r_seq[i] > c_sup:
                self.concepts[i][:] = -1
            elif not (which_c[0] == -1):
                for c in range(self.concepts.shape[1]):
                    if c not in which_c:
                        self.concepts[i, c] = -1

        # print(self.concepts.shape)
        self.concepts = np.stack(self.concepts, axis=0)
        self.labels = np.stack(self.labels, axis=0)
        self.list_embeddings = np.array(new_list_embeddings)

    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __getitem__(self, item):
        labels = self.labels[item]
        concepts = self.concepts[item]
        embeddings_path = self.list_embeddings[item]
        embedding = torch.load(embeddings_path, map_location=torch.device("cpu"))
        return embedding, labels, concepts

    def __len__(self):
        return len(self.list_embeddings)


if __name__ == "__main__":
    print("Hello World")

    train_data = PreSDDOIADataset("../../data/mini_boia_embeddings", "train")
    val_data = PreSDDOIADataset("../../data/mini_boia_embeddings", "val")
    test_data = PreSDDOIADataset("../../data/mini_boia_embeddings", "test")
    ood_data = PreSDDOIADataset("../../data/mini_boia_embeddings", "ood")

    embedding, label, concepts = train_data[0]
    print(embedding.shape, concepts.shape, label.shape)
    quit()
