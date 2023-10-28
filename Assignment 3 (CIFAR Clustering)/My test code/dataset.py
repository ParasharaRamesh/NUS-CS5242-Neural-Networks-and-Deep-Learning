from functools import reduce

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from device_data_loader import *
from tqdm import tqdm
import random

'''
TODO.x:
* Implement another class which gives all instances of each class for doing JS divergance and KDE later on 

'''

'''
TODO.
2. using this from {0->9} pick ucc random classes
4. from this pick {ucc} random indices 
5. from the sub_class_dataset {ucc} pick one random image and fill it at that random index
6. also fill the corresponding ucc for that
7. In the end do some transform and stuff
8. do it only for 40k

'''


class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, ucc_limit=4, batch_size=2):
        '''
        Note these are numpy arrays

        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :param x_test:
        :param y_test:
        '''
        self.num_classes = 10
        self.bag_size = 10
        self.ucc_limit = ucc_limit
        self.batch_size = batch_size

        # transforms to apply
        self.transforms = [
            # normal
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]),
            # random horizontal flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            # random vertical flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]),
            # random rotations
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ]),
            # random rotations & flips
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]),
        ]

        # converting it all into a tensor (it's not yet one hotified)
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_test = torch.from_numpy(x_test)
        self.y_test = torch.from_numpy(y_test)
        self.x_val = torch.from_numpy(x_val)
        self.y_val = torch.from_numpy(y_val)

        # create subdatasets ([class_0_imgs, class_1_imgs,... class_9_imgs])
        self.train_sub_datasets = self.create_sub_datasets(self.x_train, self.y_train)
        self.test_sub_datasets = self.create_sub_datasets(self.x_test, self.y_test)
        self.val_sub_datasets = self.create_sub_datasets(self.x_val, self.y_val)

        # create dataloaders
        self.autoencoder_test_dataloaders = [DeviceDataLoader(test_sub_dataset, self.batch_size)
                                             for test_sub_dataset in self.test_sub_datasets]
        self.ucc_train_dataloader, self.ucc_test_dataloader, self.ucc_val_dataloader = self.get_dataloaders_for_ucc()
        self.ucc_rcc_train_dataloader, self.ucc_rcc_test_dataloader, self.ucc_rcc_val_dataloader = self.get_dataloaders_for_ucc_and_rcc()

        print("Initilized all dataloaders")

    # create dataloaders
    def get_dataloaders_for_ucc(self):
        train_dataset_with_ucc, test_dataset_with_ucc, val_dataset_with_ucc = self.construct_datasets_with_ucc()
        return DeviceDataLoader(train_dataset_with_ucc, self.batch_size), \
            DeviceDataLoader(test_dataset_with_ucc, self.batch_size), \
            DeviceDataLoader(val_dataset_with_ucc, self.batch_size)

    def get_dataloaders_for_ucc_and_rcc(self):
        train_dataset_with_ucc_and_rcc, test_dataset_with_ucc_and_rcc, val_dataset_with_ucc_and_rcc = self.construct_datasets_with_ucc_and_rcc()
        return DeviceDataLoader(train_dataset_with_ucc_and_rcc, self.batch_size), \
            DeviceDataLoader(test_dataset_with_ucc_and_rcc, self.batch_size), \
            DeviceDataLoader(val_dataset_with_ucc_and_rcc, self.batch_size)

    # create sub datasets
    def create_sub_datasets(self, x, y):
        # Initialize an empty list to store the sub-datasets
        sub_datasets = []

        # Split the original dataset into 10 sub-datasets
        for class_label in range(self.num_classes):
            # Select indices for the current class
            indices = torch.where(y == class_label)[0]

            # Extract data for the current class
            x_class = x[indices]

            # Create a TensorDataset for the current class
            class_dataset = TensorDataset(x_class)

            # Append the current class dataset to the list
            sub_datasets.append(class_dataset)
        return sub_datasets

    # pick random image from ith class
    def pick_random_from_ith_sub_dataset(self, sub_datasets, i, is_eval):
        assert 0 <= i < self.num_classes
        sub_dataset = sub_datasets[i]
        sub_dataset_length = len(sub_dataset)
        random_idx = random.randint(0, sub_dataset_length - 1)
        random_img = sub_dataset[random_idx][0]
        random_img = random_img.permute((2, 0, 1))
        if not is_eval:
            random_transform = random.choice(self.transforms)
            random_img = random_transform(random_img)
        return random_img

    # construct UCC dataset
    def construct_datasets_with_ucc(self):
        train_dataset_with_ucc = self.construct_dataset_with_ucc(self.train_sub_datasets, False)
        test_dataset_with_ucc = self.construct_dataset_with_ucc(self.test_sub_datasets, True)
        val_dataset_with_ucc = self.construct_dataset_with_ucc(self.val_sub_datasets, True)

        return train_dataset_with_ucc, test_dataset_with_ucc, val_dataset_with_ucc

    def construct_dataset_with_ucc(self, sub_datasets, is_eval):
        bag_tensors = []
        ucc_tensors = []

        # calculate no of bags needed (NOTE: we are not going to pick every image here!)
        total_bags = 0
        for sub_dataset in sub_datasets:
            total_bags += len(sub_dataset)
        total_bags = total_bags // self.bag_size

        for b in tqdm(range(total_bags)):
            # this will keep picking ucc (1 -> 4) in a cyclic manner
            ucc = (b % self.ucc_limit) + 1
            bag_tensor = [None] * 10

            # you are choosing random classes of size {ucc}. Using this knowledge you have to fill the bag up.
            chosen_classes = random.sample(list(range(self.num_classes)), ucc)
            random_bag_pos = random.sample(list(range(self.bag_size)), self.bag_size)

            # fill all the values for ucc first and then fill the remaining with random sampling with replacement
            for chosen_class, bag_pos in zip(chosen_classes, random_bag_pos[:len(chosen_classes)]):
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)

            # fill bag_tensor pos by pos
            for bag_pos in random_bag_pos[len(chosen_classes):]:
                chosen_class = random.choice(chosen_classes)
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)

            bag_tensors.append(torch.stack(bag_tensor))
            ucc_tensors.append(self.one_hot(ucc, self.ucc_limit))

        return TensorDataset(
            torch.stack(bag_tensors),
            torch.stack(ucc_tensors)
        )

    # create UCC and RCC dataset
    def construct_datasets_with_ucc_and_rcc(self):
        train_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.train_sub_datasets, False)
        test_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.val_sub_datasets, True)
        val_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.test_sub_datasets, True)

        return train_dataset_with_ucc_and_rcc, test_dataset_with_ucc_and_rcc, val_dataset_with_ucc_and_rcc

    def construct_dataset_with_ucc_and_rcc(self, sub_datasets, is_eval):
        bag_tensors = []
        ucc_tensors = []
        rcc_tensors = []

        # calculate no of bags needed (NOTE: we are not going to pick every image here!)
        total_bags = 0
        for sub_dataset in sub_datasets:
            total_bags += len(sub_dataset)
        total_bags = total_bags // self.bag_size

        for b in tqdm(range(total_bags)):
            # this will keep picking ucc (1 -> 4) in a cyclic manner
            ucc = (b % self.ucc_limit) + 1
            bag_tensor = [None] * 10
            rcc_tensor = [0] * 10

            # you are choosing random classes of size {ucc}. Using this knowledge you have to fill the bag up.
            chosen_classes = random.sample(list(range(self.num_classes)), ucc)
            random_bag_pos = random.sample(list(range(self.bag_size)), self.bag_size)

            # fill all the values for ucc first and then fill the remaining with random sampling with replacement
            for chosen_class, bag_pos in zip(chosen_classes, random_bag_pos[:len(chosen_classes)]):
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)
                rcc_tensor[chosen_class] += 1

            # fill bag_tensor pos by pos
            for bag_pos in random_bag_pos[len(chosen_classes):]:
                chosen_class = random.choice(chosen_classes)
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)
                rcc_tensor[chosen_class] += 1

            bag_tensors.append(torch.stack(bag_tensor))
            ucc_tensors.append(self.one_hot(ucc, self.ucc_limit))
            rcc_tensors.append(torch.tensor(rcc_tensor))

        return TensorDataset(
            torch.stack(bag_tensors),
            torch.stack(ucc_tensors),
            torch.stack(rcc_tensors),
        )

    # util
    def one_hot(self, label, limit):
        # Create a one-hot tensor
        one_hot = torch.zeros(limit)

        # since each label is in range of [1,10] getting it to a range of [0,9]
        one_hot[label - 1] = 1
        return one_hot


if __name__ == '__main__':
    splitted_dataset = np.load('../Dataset/splitted_cifar10_dataset.npz')

    x_train = splitted_dataset['x_train']
    y_train = splitted_dataset['y_train']
    x_val = splitted_dataset['x_val']
    y_val = splitted_dataset['y_val']
    x_test = splitted_dataset['x_test']
    y_test = splitted_dataset['y_test']

    dataset = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)
