import random
from itertools import combinations
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from params import *

class RCCDataset(Dataset):
    def __init__(self, x, y, num_iter, train_mode):
        self.x = x
        self.y = y
        self.num_iter = num_iter
        self.train_mode = train_mode

        self.bag_size = config.bag_size
        self.num_classes = config.num_classes
        self.ucc_limit = config.ucc_limit

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
        return self.num_iter * config.batch_size

    def get_class_label_to_img_idxs_dict(self):
        class_label_to_img_idxs_dict = dict()
        for label_value in range(self.num_classes):
            label_key = f"class{label_value}"

            img_idxs = np.where(self.y == label_value)[0]
            class_label_to_img_idxs_dict[label_key] = img_idxs

        return class_label_to_img_idxs_dict

    def get_ucc_to_all_combinations_dict(self):
        class_labels = np.arange(self.num_classes)
        ucc_to_all_combos_dict = dict()

        for ucc in range(self.ucc_limit):  # go from 1->4
            ucc_key = f"ucc{ucc}"

            ucc_bag_labels = list()
            for unique_ucc_combo in combinations(class_labels, ucc+1): #plus 1 here
                ucc_bag_labels.append(np.array(unique_ucc_combo))

            ucc_to_all_combos_dict[ucc_key] = np.array(ucc_bag_labels)

        return ucc_to_all_combos_dict

    def __getitem__(self, index):
        ucc_label = index % self.ucc_limit
        rcc_label = torch.zeros(self.num_classes)
        ucc_combo_with_bag_counts = self.get_random_ucc_combo_and_its_bag_counts(ucc_label)

        selected_idxs = []
        for label, freq in ucc_combo_with_bag_counts:
            label_key = f"class{label}"
            label_img_idxs = self.class_label_to_img_idxs[label_key]
            selected_idxs.extend(list(label_img_idxs[np.random.randint(0, len(label_img_idxs), size=freq)]))
            #construct the rcc label
            rcc_label[label] = freq

        imgs = self.get_imgs_from_idxs(selected_idxs)
        return imgs, ucc_label, rcc_label

    def get_random_ucc_combo_and_its_bag_counts(self, ucc_label):
        class_key = f"ucc{ucc_label}"

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

    def get_imgs_from_idxs(self, idxs):
        imgs = self.x[idxs]
        res = []
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i])
            random_transform = random.choice(self.transforms)
            res.append(random_transform(img).unsqueeze(0))
        res = torch.concatenate(res, dim=0)
        return res

if __name__ == '__main__':
    data = np.load(config.datasets_path)
    x_train, y_train = data["x_train"], data["y_train"]
    train_dataset = RCCDataset(x_train, y_train, num_iter=10, train_mode=True)
    inp1, label1, r1 = train_dataset[0]
    inp2, label2, r2 = train_dataset[1]
    inp3, label3, r3 = train_dataset[2]
    inp4, label4, r4 = train_dataset[3]
    print("done test")
