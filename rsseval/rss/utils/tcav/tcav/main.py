from torchvision import transforms
from tcav import TCAV
import torch
from model_wrapper import ModelWrapper
from mydata import MyDataset
import os
from argparse import Namespace
from pad import PadCoinToss
from collections import OrderedDict
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

from datasets.boia import BOIA
from datasets.miniboia import MINIBOIA
from datasets.minikandinsky import MiniKandinsky
from datasets.kandinsky import Kandinsky
from datasets.shortcutmnist import SHORTMNIST
from datasets.clipshortcutmnist import CLIPSHORTMNIST
from datasets.clipboia import CLIPBOIA
from datasets.clipminiboia import CLIPMiniboia
from datasets.clipkandinsky import CLIPKandinsky
from models.boiann import BOIAnn
from models.miniboiann import MiniBOIAnn
from models.miniboiacbm import MiniBoiaCBM
from models.boiacbm import BoiaCBM
from models.mnistcbm import MnistCBM
from models.mnistnn import MNISTnn
from models.kandnn import KANDnn


def data_loader(base_path, dataset_name):
    data_transforms = data_transforms = transforms.Compose([transforms.ToTensor()])
    embedding = False

    if dataset_name == "shortmnist":
        data_transforms = transforms.Compose(
            [PadCoinToss(56), transforms.Grayscale(), transforms.ToTensor()]
        )

    if dataset_name in [
        "boia",
        "clipshortmnist",
        "clipminiboia",
        "clipboia",
        "clipkandinsky",
    ]:
        embedding = True

    image_dataset_train = MyDataset(
        base_path, transform=data_transforms, embedding=embedding
    )
    train_loader = DataLoader(image_dataset_train, batch_size=1, num_workers=0)
    return train_loader


def validate(
    model, dataset_name, validloader, concept_dict, class_dict, seed, model_name, add=""
):
    extract_layer = None
    is_boia = True

    if dataset_name in ["shortmnist", "clipshortmnist"]:
        extract_layer = "conv2"  # conv1, conv2, fc1, fc2
        is_boia = False
    if dataset_name in ["boia", "clipboia"]:
        extract_layer = "fc1"  # fc1, fc2, fc3, fc4
    if dataset_name in ["miniboia", "clipminiboia"]:
        extract_layer = "fc2"  # conv1, conv2, conv3, conv4, conv5, conv6, fc1, fc2
    if dataset_name in ["kandinsky", "minikandinsky", "clipkandinsky"]:
        extract_layer = "fc2"

    model = ModelWrapper(model, [extract_layer], is_boia)
    scorer = TCAV(model, validloader, concept_dict, class_dict.values(), 150, is_boia)

    print("Generating concepts...")
    scorer.generate_activations([extract_layer])
    scorer.load_activations()
    print("Concepts successfully generated and loaded!")

    print("Calculating TCAV scores...")
    scorer.generate_cavs(extract_layer)
    scorer.calculate_concept_presence(
        extract_layer,
        f"output/concept_presence_{dataset_name}_{model_name}_{seed}_{extract_layer}{add}.npy",
    )
    print(
        f"Done! output/concept_presence_{dataset_name}_{model_name}_{seed}_{extract_layer}{add}.npy"
    )


def get_model(modelname, encoder, args):
    if modelname.lower() == "boiann":
        return BOIAnn(encoder=encoder, args=args)
    if modelname.lower() == "miniboiann":
        return MiniBOIAnn(encoder=encoder, args=args)
    if modelname.lower() == "boiacbm":
        return BoiaCBM(encoder=encoder, args=args)
    if modelname.lower() == "miniboiacbm":
        return MiniBoiaCBM(encoder=encoder, args=args)
    if modelname.lower() == "mnistnn":
        return MNISTnn(encoder=encoder, args=args)
    if modelname.lower() == "mnistcbm":
        return MnistCBM(encoder=encoder, args=args)
    if modelname.lower() == "kandnn":
        return KANDnn(encoder=encoder, args=args)

    raise NotImplementedError(f"Model {modelname} missing")


def get_dataset(datasetname, args):
    if datasetname.lower() == "boia":
        return BOIA(args)
    if datasetname.lower() == "miniboia":
        return MINIBOIA(args)
    if datasetname.lower() == "minikandinsky":
        return MiniKandinsky(args)
    if datasetname.lower() == "kandinsky":
        return Kandinsky(args)
    if datasetname.lower() == "shortmnist":
        return SHORTMNIST(args)
    if datasetname.lower() == "clipshortmnist":
        return CLIPSHORTMNIST(args)
    if datasetname.lower() == "clipkandinsky":
        return CLIPKandinsky(args)
    if datasetname.lower() == "clipminiboia":
        return CLIPMiniboia(args)
    if datasetname.lower() == "clipboia":
        return CLIPBOIA(args)

    raise NotImplementedError(f"Dataset {datasetname} missing")


def setup():
    args = Namespace(
        backbone="neural",  # "conceptizer",
        preprocess=0,
        finetuning=0,
        batch_size=1,
        n_epochs=20,
        validate=1,
        dataset="clipminiboia",
        lr=0.001,
        exp_decay=0.99,
        warmup_steps=1,
        wandb=None,
        task="boia",
        boia_model="ce",
        model="miniboiann",
        c_sup=0,
        which_c=-1,
        joint=True,
        boia_ood_knowledge=False,
    )

    # get dataset
    dataset = get_dataset(args.dataset, args)
    # get model
    model = get_model(
        modelname=args.model, encoder=dataset.get_backbone()[0], args=args
    )

    # set cpu for the moment
    model.device = "cuda"

    model.to(model.device)
    if hasattr(model, "encoder"):
        model.encoder.to(model.device)
    if hasattr(model, "net"):
        model.net.to(model.device)

    return args, dataset, model


def mnist_tcav_setup():
    class_dict = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eigth": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
    }

    _, _, validloader = dataset.get_data_loaders()

    concepts_order = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    tmp_concept_dict = {}
    for dirname in os.listdir("../data/concepts"):
        fullpath = os.path.join("../data/concepts", dirname)
        if os.path.isdir(fullpath):
            tmp_concept_dict[dirname] = data_loader(fullpath, args.dataset)

    concept_dict = OrderedDict()
    for c in concepts_order:
        concept_dict[c] = tmp_concept_dict[c]

    return validloader, class_dict, concept_dict


def kand_tcav_setup(is_clip=False):
    class_dict = {
        "false": 0,
        "true": 1,
    }

    _, _, validloader = dataset.get_data_loaders()

    concepts_order = [
        "square",
        "triangle",
        "circle",
        "red",
        "yellow",
        "blue",
    ]

    tmp_concept_dict = {}

    lmao_name = "../data/kand-tcav/"
    if is_clip:
        lmao_name = "../data/kand-tcav-clip/"

    for dirname in os.listdir(lmao_name):
        fullpath = os.path.join(lmao_name, dirname)
        if os.path.isdir(fullpath):
            tmp_concept_dict[dirname] = data_loader(fullpath, args.dataset)

    concept_dict = OrderedDict()
    for c in concepts_order:
        concept_dict[c] = tmp_concept_dict[c]

    return validloader, class_dict, concept_dict


def boia_tcav_setup():
    class_dict = {
        "forward": int("".join(map(str, [1, 0, 0, 0])), 2),
        "stop": int("".join(map(str, [0, 1, 0, 0])), 2),
        "f_left_right": int("".join(map(str, [1, 0, 1, 1])), 2),
        "s_left_right": int("".join(map(str, [0, 1, 1, 1])), 2),
        "s_left": int("".join(map(str, [0, 1, 1, 0])), 2),
        "f_left": int("".join(map(str, [1, 0, 1, 0])), 2),
        "s_right": int("".join(map(str, [0, 1, 0, 1])), 2),
        "f_right": int("".join(map(str, [1, 0, 0, 1])), 2),
    }

    _, _, validloader = dataset.get_data_loaders()

    concept_dict = {}

    concepts_order = [
        "green_light",
        "follow",
        "clear",
        "red_light",
        "stop_sign",
        "car",
        "person",
        "rider",
        "other_obstacle",
        "no_left_lane",
        "left_obstacle",
        "left_solid_line",
        "right_lane",
        "right_green_light",
        "right_follow",
        "no_right_lane",
        "right_obstacle",
        "right_solid_line",
        "left_lane",
        "left_green_light",
        "left_follow",
    ]

    tmp_concept_dict = {}
    for dirname in os.listdir("../data/boia-preprocess-full/concepts/"):
        fullpath = os.path.join("../data/boia-preprocess-full/concepts/", dirname)
        if os.path.isdir(fullpath):
            tmp_concept_dict[dirname] = data_loader(fullpath, args.dataset)

    concept_dict = OrderedDict()
    for c in concepts_order:
        concept_dict[c] = tmp_concept_dict[c]

    return validloader, class_dict, concept_dict


def miniboia_tcav_setup(full=False):
    class_dict = {
        "forward": int("".join(map(str, [1, 0, 0, 0])), 2),
        "stop": int("".join(map(str, [0, 1, 0, 0])), 2),
        "f_left_right": int("".join(map(str, [1, 0, 1, 1])), 2),
        "s_left_right": int("".join(map(str, [0, 1, 1, 1])), 2),
        "s_left": int("".join(map(str, [0, 1, 1, 0])), 2),
        "f_left": int("".join(map(str, [1, 0, 1, 0])), 2),
        "s_right": int("".join(map(str, [0, 1, 0, 1])), 2),
        "f_right": int("".join(map(str, [1, 0, 0, 1])), 2),
    }

    _, _, validloader = dataset.get_data_loaders()

    concept_dict = {}

    concepts_order = [
        "green_light",
        "follow",
        "clear",
        "red_light",
        "stop_sign",
        "car",
        "person",
        "rider",
        "other_obstacle",
        "no_left_lane",
        "left_obstacle",
        "left_solid_line",
        "right_lane",
        "right_green_light",
        "right_follow",
        "no_right_lane",
        "right_obstacle",
        "right_solid_line",
        "left_lane",
        "left_green_light",
        "left_follow",
    ]

    folder_suffix = ""

    if full:
        folder_suffix = "-preprocess-full"

    tmp_concept_dict = {}
    for dirname in os.listdir(f"../data/miniboia{folder_suffix}/concepts"):
        fullpath = os.path.join(f"../data/miniboia{folder_suffix}/concepts", dirname)
        if os.path.isdir(fullpath):
            tmp_concept_dict[dirname] = data_loader(fullpath, args.dataset)

    concept_dict = OrderedDict()
    for c in concepts_order:
        concept_dict[c] = tmp_concept_dict[c]

    return validloader, class_dict, concept_dict


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # get everything
    args, dataset, model = setup()

    seeds = [123, 456, 789, 1011, 1213]
    model_path = f"best_model_{args.dataset}_{args.model}"
    miniboia_full = ""
    to_add = ""  # "_padd_random"

    for i, seed in enumerate(seeds):
        # get everything
        args, dataset, model = setup()

        print("Doing seed", seed)

        current_model_path = f"{model_path}_{seed}.pth"

        if not os.path.exists(current_model_path):
            print(f"{current_model_path} is missing...")
            raise ValueError("File not found")

        model_state_dict = torch.load(current_model_path)

        # Load the model status dict
        model.load_state_dict(model_state_dict)
        model.eval()

        is_clip = False
        if "clip" in args.dataset:
            is_clip = True

        if args.dataset in ["shortmnist", "clipshortmnist"]:
            validloader, class_dict, concept_dict = mnist_tcav_setup()
        elif args.dataset in ["boia", "clipboia"]:
            validloader, class_dict, concept_dict = boia_tcav_setup()
        elif args.dataset in ["kandinsky", "minikandinsky", "clipkandinsky"]:
            validloader, class_dict, concept_dict = kand_tcav_setup(is_clip)
        elif args.dataset in ["miniboia", "clipminiboia"]:
            if miniboia_full != "":
                to_add = "_full"
            validloader, class_dict, concept_dict = miniboia_tcav_setup()

        validate(
            model,
            args.dataset,
            validloader,
            concept_dict,
            class_dict,
            seed,
            args.model,
            add=to_add,
        )
