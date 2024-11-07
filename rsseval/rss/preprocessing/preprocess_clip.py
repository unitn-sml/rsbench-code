import torch
import os
import random
import preprocessing.utils as utils
import preprocessing.data_utils as data_utils
import preprocessing.similarity as similarity
import argparse
import datetime
import json


def preprocess_w_CLIP(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set == None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)

    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    d_test = args.dataset + "_test"

    all_sets = [d_train, d_val, d_test]
    # all_sets = [d_val, d_test]

    if args.dataset in ["sddoia", "shortcutmnist"]:
        d_ood = args.dataset + "_ood"
        all_sets.append(d_ood)

    # save activations and get save_paths
    if args.dataset == "shortcutmnist":

        for d_probe in all_sets:
            utils.save_activations(
                clip_name=args.clip_name,
                target_name=args.backbone,
                target_layers=[args.feature_layer],
                d_probe=d_probe,
                concept_set=args.concept_set,
                batch_size=args.batch_size,
                device=args.device,
                pool_mode="avg",
                save_dir=args.activation_dir,
                stance=0,
            )

            utils.save_activations(
                clip_name=args.clip_name,
                target_name=args.backbone,
                target_layers=[args.feature_layer],
                d_probe=d_probe,
                concept_set=args.concept_set,
                batch_size=args.batch_size,
                device=args.device,
                pool_mode="avg",
                save_dir=args.activation_dir,
                stance=1,
            )
    else:
        for d_probe in all_sets:
            utils.save_activations(
                clip_name=args.clip_name,
                target_name=args.backbone,
                target_layers=[args.feature_layer],
                d_probe=d_probe,
                concept_set=args.concept_set,
                batch_size=args.batch_size,
                device=args.device,
                pool_mode="avg",
                save_dir=args.activation_dir,
                stance=None,
            )
    ### END HERE PREPROCESSING


def get_args():
    parser = argparse.ArgumentParser(description="Settings for creating CBM")

    parser.add_argument("--dataset", type=str, default="sddoia")
    parser.add_argument(
        "--concept_set", type=str, default=None, help="path to concept set name"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="clip_RN50",
        help="Which pretrained model to use as backbone",
    )
    parser.add_argument(
        "--clip_name", type=str, default="ViT-B/32", help="Which CLIP model to use"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size used when saving model/CLIP activations",
    )
    parser.add_argument(
        "--saga_batch_size",
        type=int,
        default=256,
        help="Batch size used when fitting final layer",
    )
    parser.add_argument(
        "--proj_batch_size",
        type=int,
        default=50000,
        help="Batch size to use when learning projection layer",
    )

    parser.add_argument(
        "--feature_layer",
        type=str,
        default="layer4",
        help="Which layer to collect activations from. Should be the name of second to last layer in the model",
    )
    parser.add_argument(
        "--activation_dir",
        type=str,
        default="data/saved_activations",
        help="save location for backbone and CLIP activations",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/saved_models",
        help="where to save trained models",
    )
    parser.add_argument(
        "--clip_cutoff",
        type=float,
        default=0.25,
        help="concepts with smaller top5 clip activation will be deleted",
    )
    parser.add_argument(
        "--proj_steps",
        type=int,
        default=1000,
        help="how many steps to train the projection layer for",
    )
    parser.add_argument(
        "--interpretability_cutoff",
        type=float,
        default=0.45,
        help="concepts with smaller similarity to target concept will be deleted",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.0007,
        help="Sparsity regularization parameter, higher->more sparse",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=1000,
        help="How many iterations to run the final layer solver for",
    )
    parser.add_argument(
        "--print",
        type=bool,
        default=True,
        help="Print all concepts being deleted in this stage",
    )

    args = parser.parse_args()

    return args
