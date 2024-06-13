import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np, joblib, glob
import itertools
from PIL import Image

from torchvision.datasets.folder import pil_loader

import re

import re


def clean_and_sort_filenames(filenames):
    def preprocess_filename(filename):
        # Remove _1, _2, _3 from the filename
        cleaned_filename = re.sub(r"_[123](?=\..+)", "", filename)
        return cleaned_filename

    def extract_number_for_sorting(filename):
        # Extract the number before .png for sorting
        match = re.search(r"(\d+)(?=\..+)", filename)
        if match:
            return int(match.group(1))
        else:
            return 0

    # Use a set to remove duplicates
    unique_filenames = set()
    processed_filenames = []

    for f in filenames:
        cleaned_filename = preprocess_filename(f)
        # Add only unique filenames to the processed list
        if cleaned_filename not in unique_filenames:
            processed_filenames.append(cleaned_filename)
            unique_filenames.add(cleaned_filename)

    # Sort filenames based on the extracted number
    sorted_filenames = sorted(processed_filenames, key=extract_number_for_sorting)

    return sorted_filenames


class KAND_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, preprocess=False, finetuning=0):

        # path and train/val/test type
        self.base_path = base_path
        self.split = split

        # Add args for preprocessing / finetuning
        self.finetuning = finetuning
        self.preprocess = preprocess

        # collecting images
        self.list_images = glob.glob(
            os.path.join(self.base_path, self.split, "images", "*")
        )
        self.img_number = [i for i in range(len(self.list_images))]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.concept_mask = np.array([False] * len(self.list_images))

        self.labels, self.concepts = [], []
        for item in range(len(self.list_images)):
            target_id = os.path.join(
                self.base_path,
                self.split,
                "meta",
                str(self.img_number[item]).zfill(5) + ".joblib",
            )
            meta = joblib.load(target_id)

            label = meta["y"]
            concepts, labels = [], []
            for i in range(3):
                concept = meta["fig" + str(i)]["c"][:2]
                concepts.append(concept)

                y = meta["fig" + str(i)]["y"]
                y = 3 * y[0] + y[1]
                labels.append(y)

            labels.append(label)
            labels = np.array(labels).reshape(1, -1)
            self.labels.append(labels)

            concepts = np.concatenate(concepts, axis=0).reshape(1, -1, 6)
            self.concepts.append(concepts)

        self.concepts = np.concatenate(self.concepts, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # self.metas=[]
        # for item in range(len(self.list_images)):
        #     target_id=os.path.join(self.base_path,self.split,"meta",str(self.img_number[item]).zfill(5)+".joblib")

        #     meta=joblib.load(target_id)
        #     self.metas.append(meta)

    def mask_concepts(self, cond):
        start = self.finetuning * 100
        if start > 0:
            print("Activate finetuning on ", start, "elements of the training set")
        if cond == "red":
            self.concepts[start:, :, :3] = -1
            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

        if cond == "red-and-squares":
            for i, j in itertools.product(range(3), range(3)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

    def __getitem__(self, item):
        # meta=self.metas[item]
        # label=meta["y"]
        # labels, concepts = [], []
        # for i in range(3):
        #     concept = meta['fig'+str(i)]['c'][:2]
        #     concepts.append( concept )
        #     y = meta['fig'+str(i)]['y']
        #     y = 3*y[0]+y[1]
        #     labels.append(y)
        # labels.append(label)
        # concepts = np.concatenate(concepts, axis=0).reshape(-1, 6)
        # # concepts[:, 3:] = -1
        # labels = np.array(labels)
        # concepts = torch.from_numpy(concepts)

        labels = self.labels[item]
        concepts = self.concepts[item]

        img_id = self.img_number[item]
        image_id = os.path.join(
            self.base_path, self.split, "images", str(img_id).zfill(5) + ".png"
        )
        image = pil_loader(image_id)

        if not self.preprocess:
            return self.transform(image), labels, concepts
        else:
            return img_id, self.transform(image), labels, concepts

    def __len__(self):
        return len(self.list_images)


class PreKAND_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, preprocess=False, finetuning=0):
        self.base_path = base_path
        self.split = split

        self.finetuning = finetuning

        self.list_images = glob.glob(
            os.path.join(self.base_path, self.split, "images", "*")
        )
        self.img_number = [i for i in range(len(self.list_images))]

        self.concept_mask = np.array([False] * len(self.list_images))
        self.imgs, self.labels, self.concepts = [], [], []

        # self.targets=torch.LongTensor([])

        for item in range(len(self.list_images)):
            img_id = os.path.join(
                self.base_path, self.split, "images", str(item).zfill(5) + ".npy"
            )
            tgt_id = os.path.join(
                self.base_path, self.split, "labels", str(item).zfill(5) + ".npy"
            )
            cnp_id = os.path.join(
                self.base_path, self.split, "concepts", str(item).zfill(5) + ".npy"
            )

            self.imgs.append(np.load(img_id))
            self.labels.append(np.load(tgt_id))
            self.concepts.append(np.load(cnp_id))

        self.imgs = np.concatenate(self.imgs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.concepts = np.concatenate(self.concepts, axis=0)

    def mask_concepts(self, cond):
        start = self.finetuning * 100
        if start > 0:
            print("Activate finetuning on ", start, "elements of the training set")
        if cond == "red":
            self.concepts[start:, :, :3] = -1
            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

        if cond == "red-and-squares":
            for i, j in itertools.product(range(3), range(3)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

            # for i,j in itertools.product(range(3), range(3,6)):
            #     mask = (self.concepts[:, i, j] != 0)
            #     self.concepts[:, i, j][mask] = -1

    def __getitem__(self, item):
        embs = self.imgs[item].reshape(-1)
        labels = self.labels[item].reshape(-1)
        concepts = self.concepts[item].reshape(3, -1)

        return embs, labels, concepts

    def __len__(self):
        return len(self.list_images)


class miniKAND_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, preprocess=False, finetuning=0):

        # path and train/val/test type
        self.base_path = base_path
        self.split = split

        # Add args for preprocessing / finetuning
        self.finetuning = finetuning
        self.preprocess = preprocess

        # collecting images
        self.list_images = glob.glob(os.path.join(self.base_path, self.split, "*"))
        self.list_images = list(sorted(self.list_images))

        self.img_number = [i for i in range(len(self.list_images))]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.concept_mask = np.array([False] * len(self.list_images))

        self.labels, self.concepts = [], []
        for item in range(len(self.list_images)):
            target_id = os.path.join(
                self.base_path,
                self.split + "_meta",
                str(self.img_number[item]).zfill(5) + ".joblib",
            )
            meta = joblib.load(target_id)

            label = meta["y"]
            concepts, labels = [], []
            for i in range(3):
                concept = meta["fig" + str(i)]["c"][:2]
                concepts.append(concept)

                y = meta["fig" + str(i)]["y"]
                y = 3 * (y[0]) + y[1]
                labels.append(y)

            labels.append(label)
            labels = np.array(labels).reshape(1, -1)
            self.labels.append(labels)

            concepts = np.concatenate(concepts, axis=0).reshape(1, -1, 6)
            self.concepts.append(concepts)

        self.concepts = np.concatenate(self.concepts, axis=0)
        import copy

        self.original_concepts = copy.deepcopy(self.concepts)
        self.labels = np.concatenate(self.labels, axis=0)

    def mask_concepts(self, cond, obj=None):
        start = self.finetuning
        if start > 0:
            print("Activate finetuning on ", start, "elements of the training set")

        assert obj is None or (obj >= 0 and obj <= 8)

        if obj is not None:
            self.concepts[start:] = -1
            print("IL BOIAZZA")
            n_figure = obj // self.concepts.shape[1]
            n_obj = obj % self.concepts.shape[1]
            for i, j in itertools.product(range(3), range(6)):
                if i == n_figure and (
                    j == n_obj or j == (n_obj + self.concepts.shape[2] // 2)
                ):
                    print("Il boia colpisce", self.concepts[:start, :, :].shape)
                    print("La vittima è", self.concepts[:start, i, j])
                    pass
                else:
                    self.concepts[:start, i, j] = -1

        elif cond == "red-square":
            # self.concepts[start:,:,:3] = -1
            for i, j in itertools.product(range(3), range(3)):
                mask1 = self.concepts[:, i, j] == 0
                mask2 = self.concepts[:, i, 3 + j] == 0

                rs_mask = mask1 & mask2

                count = 0
                for l in range(len(rs_mask)):
                    if count == 10:
                        rs_mask[l:] = False
                        break
                    elif rs_mask[l]:
                        count += 1
                self.concepts[:, i, j][~rs_mask] = -1
                self.concepts[:, i, 3 + j][~rs_mask] = -1

        elif cond == "red":
            self.concepts[start:, :, :3] = -1
            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

        elif cond == "red-and-squares":
            for i, j in itertools.product(range(3), range(3)):
                mask = self.concepts[:20, i, j] != 0
                self.concepts[:20, i, j][mask] = -1

            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[:20, i, j] != 0
                self.concepts[:20, i, j][mask] = -1
        elif cond == "red-and-squares-and-circle":
            for i, j in itertools.product(range(3), range(3)):
                mask_1 = self.concepts[:20, i, j] != 0
                mask_2 = self.concepts[:20, i, j] != 1
                mask = mask_1 | mask_2
                self.concepts[:20, i, j][mask] = -1

            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[:20, i, j] != 0
                self.concepts[:20, i, j][mask] = -1
        else:
            self.concepts[start:] = -1

    def mask_concepts_specific(self, data_idx, figure_idx, obj_idx):

        mask = np.isin(np.arange(self.concepts.shape[0]), data_idx)
        mask_expanded = mask[:, np.newaxis, np.newaxis]
        self.concepts = np.where(mask_expanded, self.concepts, -1)

        for idx, fig, obj in zip(data_idx, figure_idx, obj_idx):
            for i, j in itertools.product(range(3), range(6)):
                if i == fig and (j == obj or j == obj + 3):
                    pass
                else:
                    self.concepts[idx, i, j] = -1

    def __getitem__(self, item):
        labels = self.labels[item]
        concepts = self.concepts[item]

        img_id = self.img_number[item]
        all_imgs = []
        for i in range(9):
            image_id = os.path.join(
                self.base_path,
                self.split,
                str(img_id).zfill(5),
                str(i).zfill(5) + ".png",
            )
            image = pil_loader(image_id)

            # image = Image.open(image_id)
            # image = np.array(image)

            t_image = self.transform(image)

            # if item==0:
            #     new_image = Image.fromarray((t_image.permute(2,1,0).numpy()*255).astype(np.uint8))
            #     new_image.save(f"../../data/new_image{i}.png")

            all_imgs.append(t_image)

        return torch.cat(all_imgs, dim=-1), labels, concepts

    def __len__(self):
        return len(self.list_images)


class CLIP_KAND_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, preprocess=False, finetuning=0):
        self.base_path = base_path
        self.split = split

        self.finetuning = finetuning

        list_images = glob.glob(
            os.path.join(self.base_path, "kand-3k", self.split, "*")
        )

        self.list_images = clean_and_sort_filenames(list_images)

        self.img_number = [i for i in range(len(self.list_images))]

        self.concept_mask = np.array([False] * len(self.list_images))

        # self.targets=torch.LongTensor([])

        self.labels, self.concepts = [], []
        for item in range(len(self.list_images)):
            target_id = os.path.join(
                self.base_path,
                "kand-3k",
                self.split + "_meta",
                str(self.img_number[item]).zfill(5) + ".joblib",
            )
            meta = joblib.load(target_id)

            label = meta["y"]
            concepts, labels = [], []
            for i in range(3):
                concept = meta["fig" + str(i)]["c"][:2]
                concepts.append(concept)

                y = meta["fig" + str(i)]["y"]
                y = 3 * y[0] + y[1]
                labels.append(y)

            labels.append(label)
            labels = np.array(labels).reshape(1, -1)
            self.labels.append(labels)

            concepts = np.concatenate(concepts, axis=0).reshape(1, -1, 6)
            self.concepts.append(concepts)

        self.concepts = np.concatenate(self.concepts, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        IMG_PATH = os.path.join(
            self.base_path,
            "saved_activations",
            f"kandinsky_{self.split}_clip_ViT-B32.pt",
        )
        self.embs = torch.load(IMG_PATH)

        TXT_PATH = os.path.join(
            self.base_path, "saved_activations", "kandinsky_filtered_ViT-B32.pt"
        )
        self.texts = torch.load(TXT_PATH)

        self.imgs = self.embs  # .to('cuda') @ self.texts.T.to('cuda')
        # self.imgs = self.embs.to('cuda') # @ self.texts.T.to('cuda')

        self.imgs = self.imgs.to(torch.float64).detach().cpu().numpy()

        # print(self.imgs.shape)
        # print(self.imgs[0])
        # quit()

        # self.imgs = self.imgs.reshape(-1,3,18)
        self.imgs = self.imgs.reshape(-1, 3 * 512)

    def mask_concepts(self, cond):
        start = self.finetuning * 100
        if start > 0:
            print("Activate finetuning on ", start, "elements of the training set")
        if cond == "red":
            self.concepts[start:, :, :3] = -1
            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

        if cond == "red-and-squares":
            for i, j in itertools.product(range(3), range(3)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

            for i, j in itertools.product(range(3), range(3, 6)):
                mask = self.concepts[start:, i, j] != 0
                self.concepts[start:, i, j][mask] = -1

            # for i,j in itertools.product(range(3), range(3,6)):
            #     mask = (self.concepts[:, i, j] != 0)
            #     self.concepts[:, i, j][mask] = -1

    def __getitem__(self, item):
        embs = self.imgs[item].reshape(-1)
        labels = self.labels[item].reshape(-1)
        concepts = self.concepts[item].reshape(3, -1)

        return embs, labels, concepts

    def __len__(self):
        return len(self.list_images)


# if __name__=='__main__':
#     train_data = PreKAND_Dataset('../../data/kand-preprocess', 'train')
#     print(len(train_data))

#     print(train_data[0][2].shape,' ', train_data[0][1].shape )


#     for i in range(len(train_data)):
#         pass
#         # print(train_data[i][2],'->', train_data[i][1])

if __name__ == "__main__":

    train_data = CLIP_KAND_Dataset("../../data/", "train")

    print(len(train_data))

    val_data = CLIP_KAND_Dataset("../../data/", "val")
    print(len(val_data))

    test_data = CLIP_KAND_Dataset("../../data/", "test")
    print(len(test_data))

    img, label, concepts = train_data[:2]
    print(img.shape, concepts.shape, label.shape)

    import matplotlib.pyplot as plt

    imgs = []
    for i in range(len(train_data)):
        imgs.append(train_data[i][0])

    imgs = torch.stack(imgs, dim=0)
    print(imgs.shape)
    # print(concepts)
    # print(label)
    # plt.imshow(img.numpy())
    # plt.show()

    # Convert the array to uint8 (8-bit) for image representation

    quit()

    img = img.permute(1, 2, 0)
    image_array = img.numpy()
    image_array = image_array * 255
    image_array = image_array.astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(image_array, "RGB")

    # Save the image
    image.save("../../data/output_image.png")

    print(img.max())


#     for dset in [train_data]:
#         labels= []
#         for i in range(len(dset)):
#             # print(dset[i][2].reshape(-1,3))
#             labels.append(dset[i][1].reshape(-1,4))

#             # print(dset[i][2],'->', dset[i][1] )

#         labels = np.concatenate(labels, axis=0)

#         frac = np.sum(labels[:,0] == 1) / len(labels) + np.sum(labels[:,1] == 1) / len(labels) + np.sum(labels[:,2] == 1) / len(labels)
#         frac /= 3

#         print(dset.split, ' ', frac)

#         print(dset.split, ' ',  np.sum(labels[:,-1] == 1) / len(labels))

#     print(labels.shape)
