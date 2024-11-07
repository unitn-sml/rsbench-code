import torch
from torchvision import transforms
from PIL import Image
import random


class PadRight:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        width, height = img.size
        padding_right = max(0, self.target_width - width)
        padding = (0, 0, padding_right, 0)
        return transforms.functional.pad(img, padding, fill=0)


class PadLeft:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        width, height = img.size

        padding_left = max(0, self.target_width - width)
        padding = (padding_left, 0, 0, 0)
        return transforms.functional.pad(img, padding, fill=0)


class PadCoinToss:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        pad_left = random.choice([True, False])

        if pad_left:
            return PadLeft(self.target_width)(img)
        else:
            return PadRight(self.target_width)(img)

class PadRightDefine:
    def __init__(self, pad_width):
        self.pad_width = pad_width
    def __call__(self, img):
        padding = (0, 0, self.pad_width, 0)
        return transforms.functional.pad(img, padding, fill=0)

class PadLeftDefine:
    def __init__(self, pad_width):
        self.pad_width = pad_width
    def __call__(self, img):
        padding = (self.pad_width, 0, 0, 0)
        return transforms.functional.pad(img, padding, fill=0)

if __name__ == "__main__":
    target_width = 56

    transform = transforms.Compose([PadLeft(target_width), transforms.ToTensor()])

    img = Image.open("../data/concepts/0/0.png")
    padded_img = transform(img)

    print(padded_img.size())
