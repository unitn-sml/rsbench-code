import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

import sys
project_root = "/home/park1119/rsbench-code/rsseval/rss"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.shortcutmnist import SHORTMNIST
from argparse import Namespace

def generate_mnist_concepts(output_base="data/concepts/mnist"):
    """Generate concept folders for shortmnist using the strategy:
    - For concept "1", save samples where first digit is 1 (labels like 1x)
    - These will be padded differently during TCAV evaluation to isolate the concept
    """
    
    # Setup dataset
    args = Namespace(
        backbone="neural", #
        preprocess=0,
        finetuning=0,
        batch_size=32,
        n_epochs=40,
        validate=0, # validate=0 means evaluate on test, not validation (train.py)
        dataset="shortmnist",
        lr=0.001,
        exp_decay=0.99,
        warmup_steps=0, # this is default even when it was first trained
        wandb=None,
        task="addition",
        model="mnistnn",
        c_sup=1,  # Set to 1 to get actual digit labels instead of -1
        which_c=[-1],
        joint=False # True only for CLIP
    )
    
    # args = Namespace(
    #     backbone="neural",
    #     preprocess=0,
    #     finetuning=0,
    #     batch_size=1,
    #     n_epochs=40,
    #     validate=0,
    #     dataset="shortmnist",
    #     lr=0.001,
    #     exp_decay=0.99,
    #     warmup_steps=0,
    #     wandb=None,
    #     model="mnistnn"
    # )
    
    dataset = SHORTMNIST(args)
    train_loader, _, _ = dataset.get_data_loaders()
    
    # Load the RAW unfiltered data to get digit labels
    raw_data_path = os.path.join(project_root, "datasets/utils/2mnist_10digits/2mnist_10digits.pt")
    raw_data = torch.load(raw_data_path, weights_only=False)
    raw_labels = raw_data['train']['labels']  # Shape: (42000, 3) with [digit1, digit2, sum]
    print(f"Raw labels shape: {raw_labels.shape}")
    
    # Also need to load train indexes to reverse map from actual images to digit pairs
    train_indexes = torch.load(os.path.join(project_root, "datasets/utils/2mnist_10digits/train_indexes.pt"), weights_only=False)
    
    # Create a mapping from image to its digit pair
    # The indexes structure is: {(digit1, digit2): [list of indices in raw dataset]}
    idx_to_digits = {}
    for (d1, d2), indices in train_indexes.items():
        for idx in indices:
            idx_to_digits[idx.item()] = (d1, d2)
    
    # Access the dataset directly
    train_dataset = dataset.dataset_train
    
    # Create concept directories (one for each digit 0-9)
    for digit in range(10):
        concept_dir = os.path.join(project_root, output_base, str(digit))
        os.makedirs(concept_dir, exist_ok=True)
    
    # Process dataset and save images by concept (iterating directly, not via dataloader)
    img_count = {i: 0 for i in range(10)}
    
    print(f"Processing {len(train_dataset)} images...")
    print("Extracting individual digits from 2-digit images...")
    
    for idx in range(len(train_dataset)):
        # Get image and concepts directly from dataset
        img, label, concepts = train_dataset[idx]
        
        # Use concepts to get the actual digit pair for this image
        first_digit = int(concepts[0])
        second_digit = int(concepts[1])
        
        # img is shape (1, 28, 56) - grayscale, height=28, width=56 (two 28x28 digits side by side)
        # Extract left half (first digit) and right half (second digit)
        img_pil = transforms.ToPILImage()(img)  # Convert to PIL
        
        # Convert to numpy for easier slicing
        img_array = np.array(img_pil)  # Shape: (28, 56)
        
        # Extract first digit (left half, columns 0-27)
        first_digit_array = img_array[:, 0:28]
        first_digit_pil = Image.fromarray(first_digit_array.astype(np.uint8))
        
        # Extract second digit (right half, columns 28-55)
        second_digit_array = img_array[:, 28:56]
        second_digit_pil = Image.fromarray(second_digit_array.astype(np.uint8))
        
        # Save first digit to its concept folder
        concept_dir_first = os.path.join(project_root, output_base, str(first_digit))
        img_path_first = os.path.join(concept_dir_first, f"{img_count[first_digit]:06d}.png")
        first_digit_pil.save(img_path_first)
        img_count[first_digit] += 1
        
        # Save second digit to its concept folder
        concept_dir_second = os.path.join(project_root, output_base, str(second_digit))
        img_path_second = os.path.join(concept_dir_second, f"{img_count[second_digit]:06d}.png")
        second_digit_pil.save(img_path_second)
        img_count[second_digit] += 1
        
        if idx % 5000 == 0:
            print(f"Processed {idx} images, counts: {img_count}")

if __name__ == "__main__":
    generate_mnist_concepts()
    print("Concept generation complete!")
