import random
from itertools import combinations
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from params import *

class CIFARDataset(Dataset):
    def __init__(self, x, y, num_iter, train_mode):
        self.x = x
        self.y = y
        self.num_iter = num_iter
        self.train_mode = train_mode

        # pick the augmentation based on the mode
        if self.train_mode:
            self.transforms = [
                # normal
                transforms.Compose([
                    transforms.ToTensor()
                ]),
                # random horizontal flips
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ]),
                # random rotations
                transforms.Compose([
                    transforms.RandomRotation(3),
                    transforms.ToTensor()
                ]),
                # random rotations & flips
                transforms.Compose([
                    transforms.RandomRotation(3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            ]
        else:
            self.transforms = [
                transforms.Compose([
                    transforms.ToTensor(),
                ])
            ]

    def __len__(self):
        return self.num_iter * config.batch_size

    def get_img_from_idx(self, index):
        img = self.x[index]
        img = Image.fromarray(img)
        random_transform = random.choice(self.transforms)
        return random_transform(img)

    def get_label_from_idx(self, index):
        return self.y[index][0]

    def __getitem__(self, index):
        index = index % len(self.x)
        img = self.get_img_from_idx(index)
        label = self.get_label_from_idx(index)
        return img, label

if __name__ == '__main__':
    data = np.load(config.datasets_path)
    x_train, y_train = data["x_train"], data["y_train"]
    train_dataset = CIFARDataset(x_train, y_train, num_iter=10, train_mode=True)
    inp1, label1 = train_dataset[0]
    inp2, label2 = train_dataset[1]
    inp3, label3 = train_dataset[2]
    inp4, label4 = train_dataset[3]
    print("done test")
