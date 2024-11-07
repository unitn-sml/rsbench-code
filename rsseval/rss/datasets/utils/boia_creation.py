import torch
import torch.utils.data
from torch.utils.data import Dataset
import pickle
from datasets.utils.mnist_creation import generate_r_seq
from datasets.utils.sddoia_creation import CONCEPTS_ORDER


class BOIADataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the BDD dataset
    """

    def __init__(
        self,
        pkl_file_path,
        use_attr,
        no_img,
        uncertain_label,
        image_dir,
        n_class_attr,
        transform=None,
        c_sup=1,
        which_c=[-1],
    ):
        """
        Arguments:
        pkl_file_path: path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = "train" in pkl_file_path
        if not self.is_train:
            assert ("test" in pkl_file_path) or ("val" in pkl_file_path)
        self.data.extend(pickle.load(open(pkl_file_path, "rb")))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

        self.r_seq = generate_r_seq(len(self.data))
        self.c_sup = c_sup
        self.which_c = which_c
        # self.numel = 0

        # for i in range(self.__len__()):
        #     self.__getitem__(i)

        # print("Given supervision to", self.numel)
        # quit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]

        t_path = img_path[:-4] + ".pt"
        img_path = self.image_dir + "/inputs/" + t_path
        lab_path = self.image_dir + "/labels/" + t_path
        con_path = self.image_dir + "/concepts/" + t_path

        img = torch.load(img_path).squeeze(0)
        class_label = torch.load(lab_path).squeeze(0)
        attr_label = torch.load(con_path).squeeze(0)

        # assert class_label[4] == 0, class_label
        # NOTE: we are not interested in the last action

        # filter for the concept supervision given
        # if self.r_seq[idx] > self.c_sup:
        #     attr_label[:] = -1
        # elif not (self.which_c[0] == -1):
        #     # print("entro?")
        #     for c in range(attr_label.shape[0]):
        #         if c not in self.which_c:
        #             attr_label[c] = -1
        # else:
        #     self.numel += 1

        # filter for the concept supervision given

        if self.c_sup != 1:
            if self.r_seq[idx] > self.c_sup:
                attr_label[:] = -1
            elif not (self.which_c[0] == -1):
                # print("entro?")
                for c in range(attr_label.shape[0]):
                    if c not in self.which_c:
                        attr_label[c] = -1
            else:
                # self.numel += 1
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
                        attr_label[order] = -1

        class_label = class_label[:4]

        return img, class_label, attr_label


## --------------------------------------------------------------------------------------------------------


class CLIPBOIADataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the BDD dataset
    """

    def __init__(
        self,
        pkl_file_path,
        use_attr,
        no_img,
        uncertain_label,
        image_dir,
        n_class_attr,
        transform=None,
        c_sup=1,
        which_c=[-1],
    ):
        """
        Arguments:
        pkl_file_path: path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = "train" in pkl_file_path

        if not self.is_train:
            assert ("test" in pkl_file_path) or ("val" in pkl_file_path)
        self.data.extend(pickle.load(open(pkl_file_path, "rb")))

        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = "data/saved_activations/BOIA-preprocessed/"
        self.others_dir = image_dir
        self.n_class_attr = n_class_attr

        self.r_seq = generate_r_seq(len(self.data))
        self.c_sup = c_sup
        self.which_c = which_c

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]

        t_path = img_path[:-4] + ".pt"
        img_path = self.image_dir + "all/" + t_path
        lab_path = self.others_dir + "/labels/" + t_path
        con_path = self.others_dir + "/concepts/" + t_path

        img = torch.load(img_path).squeeze(0)
        class_label = torch.load(lab_path).squeeze(0)
        attr_label = torch.load(con_path).squeeze(0)

        # assert class_label[4] == 0, class_label
        # NOTE: we are not interested in the last action

        # filter for the concept supervision given
        if self.r_seq[idx] > self.c_sup:
            attr_label[:] = -1
        elif not (self.which_c[0] == -1):
            for c in range(attr_label.shape[0]):
                if c not in self.which_c:
                    attr_label[c] = -1

        class_label = class_label[:4]

        return img, class_label, attr_label
