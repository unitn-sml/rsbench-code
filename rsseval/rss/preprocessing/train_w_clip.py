# manage importing datasets
import torch
import os
import random
import preprocessing.utils as utils
import preprocessing.data_utils as data_utils
import preprocessing.similarity as similarity
import datetime
import json

from datasets.utils.kand_creation import KAND_Dataset

from utils.glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset


def train_LF_CBM(args):

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set == None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)

    similarity_fn = similarity.cos_similarity_cubed_single

    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"

    # get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    if args.dataset == "kandinsky":
        classes = ["false", "true"]

    target_save_name, clip_save_name, text_save_name = utils.get_save_names(
        args.clip_name,
        args.backbone,
        args.feature_layer,
        d_train,
        args.concept_set,
        "avg",
        args.activation_dir,
    )
    val_target_save_name, val_clip_save_name, text_save_name = utils.get_save_names(
        args.clip_name,
        args.backbone,
        args.feature_layer,
        d_val,
        args.concept_set,
        "avg",
        args.activation_dir,
    )

    # load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()

        val_target_features = torch.load(
            val_target_save_name, map_location="cpu"
        ).float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        # for i in range(21):
        #     for j in range(i+1,21):
        #         a = torch.dot(text_features[:,i], text_features[:,j])
        #         print(i, j)
        #         print( a.item())
        #         print()

        # for i in range(17):
        #     for j in range(i+1,18):
        #         a = torch.dot(text_features[:,i], text_features[:,j])
        #         print(i, j)
        #         print( a.item())
        #         print()

        # print(text_features.shape)

        del image_features, text_features, val_image_features

    print(concepts, "\n")

    clip_features = clip_features.view(-1, 3, 18)

    # filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i] <= args.clip_cutoff:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))

    if not args.dataset == "kandinsky":
        concepts = [
            concepts[i] for i in range(len(concepts)) if highest[i] > args.clip_cutoff
        ]

    # save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[
            highest > args.clip_cutoff
        ]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        del image_features, text_features

    if not args.dataset == "kandinsky":
        val_clip_features = val_clip_features[:, highest > args.clip_cutoff]

    # learn projection layer
    proj_layer = torch.nn.Linear(
        in_features=target_features.shape[1], out_features=len(concepts), bias=False
    ).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)

        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == args.proj_steps - 1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(
                    val_clip_features.to(args.device).detach(), val_output
                )
                val_loss = torch.mean(val_loss)
            if i == 0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print(
                    "Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(
                        best_step, -loss.cpu(), -best_val_loss.cpu()
                    )
                )

            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else:  # stop if val loss starts increasing
                break
        opt.zero_grad()

    proj_layer.load_state_dict({"weight": best_weights})
    print(
        "Best step:{}, Avg val similarity:{:.4f}".format(
            best_step, -best_val_loss.cpu()
        )
    )

    # delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff

    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i] <= args.interpretability_cutoff:
                print("Deleting {}, Interpretability:{:.3f}".format(concept, sim[i]))
    if not args.dataset == "kandinsky":
        concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]

    del clip_features, val_clip_features

    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(
        in_features=target_features.shape[1], out_features=len(concepts), bias=False
    )
    proj_layer.load_state_dict({"weight": W_c})

    ## reimplement loading of labels
    train_dataset = KAND_Dataset("data/kandinsky/data", split="train")
    val_dataset = KAND_Dataset("data/kandinsky/data", split="val")

    train_targets = [k["y"] for k in train_dataset.metas]
    val_targets = [k["y"] for k in val_dataset.metas]

    train_g = [k["c"] for k in train_dataset.metas]
    val_g = [k["c"] for k in val_dataset.metas]

    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())

        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)

        train_c -= train_mean
        train_c /= train_std

        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std

        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_c, val_y)

    indexed_train_loader = DataLoader(
        indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = args.lam

    # Solve the GLM path
    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        STEP_SIZE,
        args.n_iters,
        ALPHA,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(target_features),
        n_classes=len(classes),
    )
    W_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]

    save_name = "{}/{}_cbm_{}".format(
        args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    )
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name, "W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    with open(os.path.join(save_name, "concepts.txt"), "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)

    with open(os.path.join(save_name, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(save_name, "metrics.txt"), "w") as f:
        out_dict = {}
        for key in ("lam", "lr", "alpha", "time"):
            out_dict[key] = float(output_proj["path"][0][key])
        out_dict["metrics"] = output_proj["path"][0]["metrics"]
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total,
        }
        json.dump(out_dict, f, indent=2)
