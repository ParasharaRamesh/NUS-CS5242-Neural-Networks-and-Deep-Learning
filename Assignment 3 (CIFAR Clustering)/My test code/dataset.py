'''
TODO:
1. do the dataset the same way he did
2. use the same logic for your rcc dataset as well
3. do the same thing for the autoencoder dataloader
4. do the same thing for the kde dataloader

'''

from itertools import combinations
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from params import *


class UCCDataset(Dataset):
    def __init__(self, x, y, num_iter, train_mode):
        self.x = x
        self.y = y
        self.num_iter = num_iter
        self.train_mode = train_mode

        self.bag_size = config["bag_size"]
        self.num_classes = config["num_classes"]
        self.ucc_limit = config["ucc_limit"]

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

        # for each class get all the indexes of images
        self.class_label_to_img_idxs = self.get_class_label_to_img_idxs_dict()

        # for each ucc class from (1->4) get all the unique combinations possible
        self.ucc_to_all_combos = self.get_ucc_to_all_combinations_dict()

    def __len__(self):
        return self.num_iter * config["batch_size"]

    def get_class_label_to_img_idxs_dict(self):
        class_label_to_img_idxs_dict = dict()
        for label_value in range(self.ucc_limit):
            label_key = f"class{label_value}"

            img_idxs = np.where(self.y == label_value)[0]
            class_label_to_img_idxs_dict[label_key] = img_idxs

        return class_label_to_img_idxs_dict

    def get_ucc_to_all_combinations_dict(self):
        ucc_labels = np.arange(self.ucc_limit)
        ucc_to_all_combos_dict = dict()

        for ucc in range(1, self.ucc_limit + 1):  # go from 1->4
            ucc_key = f"ucc{ucc}"

            ucc_bag_labels = list()
            for unique_ucc_combo in combinations(ucc_labels, ucc):
                ucc_bag_labels.append(np.array(unique_ucc_combo))

            ucc_to_all_combos_dict[ucc_key] = np.array(ucc_bag_labels)

        return ucc_to_all_combos_dict

    #TODO.x from here
    def __getitem__(self, index):
        ucc_label = index % self.ucc_limit
        label_freq_list = self.get_instances_per_label_in_bag(ucc_label)

        selected_idxs = []
        for label, freq in label_freq_list:
            label_key = f"label{label}"
            img_idxs = self.class_label_to_img_idxs[label_key]
            N = len(img_idxs)
            selected_idxs += list(img_idxs[np.random.randint(0, N, size=freq)])

        inp = self.preprocess_inputs(selected_idxs)
        if self.return_classes:
            assert len(label_freq_list) == 1
            return inp, label_freq_list[0][0]
        else:
            return inp, ucc_label

    def get_instances_per_label_in_bag(self, ucc_label):
        class_key = f"class_{ucc_label}"

        # Get unique combination of cifar10 labels for ucc label
        ucc_bag_labels_list = self.ucc_to_all_combos[class_key]
        idx = np.random.randint(0, ucc_bag_labels_list.shape[0])
        ucc_labels = ucc_bag_labels_list[idx, :]

        # Get even distribution of instances per label with max difference of 1
        N = ucc_labels.shape[0]
        n_instances = self.bag_size // N

        counts = np.repeat(n_instances, N)

        res = []
        for label, freq in zip(ucc_labels, counts):
            res.append((label, freq))
        return res

    def preprocess_inputs(self, idxs):
        imgs = self.x[idxs]
        res = []
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i])
            res.append(self.transforms(img).unsqueeze(0))
        res = torch.concatenate(res, dim=0)
        return res

# TODO.1 dataloader function for ucc
# TODO.2 dataloader function for rcc
# TODO.3 dataloader function for kde
# TODO.4 dataloader function for autoencoder ( can be the same thing)
