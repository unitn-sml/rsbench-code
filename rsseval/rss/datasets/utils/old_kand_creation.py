import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np, joblib, glob

from torchvision.datasets.folder import pil_loader


class KAND_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split):
        self.base_path = base_path
        self.split = split

        self.list_images = glob.glob(
            os.path.join(self.base_path, self.split, "images", "*")
        )
        self.img_number = [i for i in range(len(self.list_images))]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.concept_mask = np.array([False] * len(self.list_images))
        self.metas = []
        self.targets = torch.LongTensor([])
        for item in range(len(self.list_images)):
            target_id = os.path.join(
                self.base_path,
                self.split,
                "meta",
                str(self.img_number[item]).zfill(5) + ".joblib",
            )
            meta = joblib.load(target_id)
            self.metas.append(meta)

    def __getitem__(self, item):
        meta = self.metas[item]
        label = meta["y"]

        labels, concepts = [], []
        y_final = 1
        for i in range(3):
            concept = meta["fig" + str(i)]["c"][:2]
            concepts.append(concept)

            # can you learn a red triangle?
            y = logic_triangle_circle(concept)
            labels.append(y)

            # labels.append( meta['fig'+str(i)]['y'])
            y_final *= labels[i]

        labels.append(y_final)
        # labels.append(label)

        concepts = np.concatenate(concepts, axis=0).reshape(-1, 8)
        labels = np.array(labels)
        # concepts = torch.from_numpy(concepts)

        img_id = self.img_number[item]
        image_id = os.path.join(
            self.base_path, self.split, "images", str(img_id).zfill(5) + ".png"
        )
        image = pil_loader(image_id)

        return self.transform(image), labels, concepts

    def __len__(self):
        return len(self.list_images)


def logic_red_triangle(concepts):

    shapes = concepts[0]
    colors = concepts[1]

    red_triangle = 0
    for i in range(4):
        if shapes[i] == 0 and colors[i] == 0:
            red_triangle = 1
            break
    return red_triangle


def logic_triangle_circle(concepts):

    shapes = concepts[0]
    colors = concepts[1]

    rt, yc = [], []
    for i in range(4):
        rt.append((shapes[i] == 0) and (colors[i] == 0))
        yc.append((shapes[i] == 1) and (colors[i] == 1))

    rt = np.min((1, np.sum(rt)))
    yc = np.min((1, np.sum(yc)))

    y = rt * yc

    return y


def logic_triangle_circle(concepts):

    shapes = concepts[0]
    colors = concepts[1]

    rt, yc = [], []
    for i in range(4):
        rt.append((shapes[i] == 0) and (colors[i] == 0))
        yc.append((shapes[i] == 1) and (colors[i] == 1))

    rt = np.min((1, np.sum(rt)))
    yc = np.min((1, np.sum(yc)))

    y = rt * yc

    return y


if __name__ == "__main__":

    train_data = KAND_Dataset("../../data/kandinsky-30k", "train")
    print(len(train_data))

    val_data = KAND_Dataset("../../data/kandinsky-30k", "val")
    print(len(val_data))

    test_data = KAND_Dataset("../../data/kandinsky-30k", "test")
    print(len(test_data))

    # img, label, concepts = test_data[0]
    # print(concepts, label)

    for dset in [train_data]:
        labels = []
        for i in range(len(dset)):
            # print(dset[i][1].reshape(-1,4))
            labels.append(dset[i][1].reshape(-1, 4))

            print(dset[i][2], "->", dset[i][1][:-1])

        labels = np.concatenate(labels, axis=0)

        frac = (
            np.sum(labels[:, 0] == 1) / len(labels)
            + np.sum(labels[:, 1] == 1) / len(labels)
            + np.sum(labels[:, 2] == 1) / len(labels)
        )
        frac /= 3

        print(dset.split, " ", frac)

        print(dset.split, " ", np.sum(labels[:, -1] == 1) / len(labels))

    print(labels.shape)


# class OOD_CLEVR(torch.utils.data.Dataset):
#     def __init__(self, base_path):

#         self.base_path = base_path

#         self.list_images= glob.glob(os.path.join(self.base_path,"image","*"))
#         self.task_number = [0] * len(self.list_images)
#         self.img_number = [i for i in range(len(self.list_images))]
#         self.transform = transforms.Compose(
#             [transforms.ToTensor()]
#         )
#         self.concept_mask=np.array([False for i in range(len(self.list_images))])
#         self.metas=[]
#         self.targets=torch.LongTensor([])
#         for item in range(len(self.list_images)):
#             target_id=os.path.join(self.base_path,"meta",str(self.img_number[item])+".joblib")
#             meta=joblib.load(target_id)
#             self.metas.append(meta)

#     @property
#     def images_folder(self):
#         return os.path.join(self.base_path,"image")

#     @property
#     def scenes_path(self):
#         return os.path.join(self.base_path,"image")

#     def __getitem__(self, item):
#         meta=self.metas[item]
#         label=meta["target"]
#         concepts= meta["concepts"]
#         mask= self.concept_mask[item]
#         if mask:
#             concepts=-torch.ones_like(concepts)
#         task_id, img_id = self.task_number[item], self.img_number[item]
#         image_id=os.path.join(self.base_path,"image",str(img_id)+".jpg")
#         image = pil_loader(image_id)
#         return self.transform(image),label,concepts,self.transform(image)

#     def __len__(self):
#         return len(self.list_images)
