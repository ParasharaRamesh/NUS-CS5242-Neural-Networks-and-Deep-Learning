import numpy as np
import torch
from torch.utils.data import TensorDataset
from device_data_loader import *
from tqdm import tqdm
import random

'''
TODO.x:
* Implement another class which gives all instances of each class for doing JS divergance and KDE later on 

'''

'''
TODO.
1. pick a random ucc {1->4}
2. using this from {0->9} pick ucc random classes
3. create a bag of size 10 init
4. from this pick {ucc} random indices 
5. from the sub_class_dataset {ucc} pick one random image and fill it at that random index
6. also fill the corresponding ucc for that
7. In the end do some transform and stuff
8. do it only for 40k

'''


class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
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

        #test pick random image
        chumma = self.pick_random_from_ith_sub_dataset(self.train_sub_datasets, 4)
        print("done")

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
    def pick_random_from_ith_sub_dataset(self, sub_datasets, i):
        assert 0 <= i < self.num_classes
        sub_dataset = sub_datasets[i]
        sub_dataset_length = len(sub_dataset)
        random_idx = random.randint(0, sub_dataset_length-1)
        return sub_dataset[random_idx][0]

    # get UCC and RCC dataset
    def construct_datasets_with_ucc(self):
        train_dataset_with_ucc = self.construct_dataset_with_ucc(self.train_loader)
        val_dataset_with_ucc = self.construct_dataset_with_ucc(self.val_loader)
        test_dataset_with_ucc = self.construct_dataset_with_ucc(self.test_loader)

        return train_dataset_with_ucc, val_dataset_with_ucc, test_dataset_with_ucc

    # create UCC and RCC dataset
    def construct_datasets_with_ucc_and_rcc(self):
        train_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.train_loader)
        val_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.val_loader)
        test_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.test_loader)

        return train_dataset_with_ucc_and_rcc, val_dataset_with_ucc_and_rcc, test_dataset_with_ucc_and_rcc

    # aux functions
    # get ucc
    def construct_dataset_with_ucc(self, dataloader):
        image_tensors = []
        ucc_tensors = []

        for data in tqdm(dataloader):
            image, label = data

        return TensorDataset(
            torch.stack(image_tensors),
            torch.stack(ucc_tensors)
        )

    # get both UCC and RCC
    def construct_dataset_with_ucc_and_rcc(self, dataloader):
        image_tensors = []
        ucc_tensors = []
        rcc_tensors = []

        for data in tqdm(dataloader):
            images, labels = data

            # get ucc
            ucc = self.get_ucc_from_labels_of_batch(labels)

            # get rcc
            rcc = self.get_rcc_from_labels_of_batch(labels)

            image_tensors.append(images)
            ucc_tensors.append(ucc)
            rcc_tensors.append(rcc)

        return TensorDataset(
            torch.stack(image_tensors),
            torch.stack(ucc_tensors),
            torch.stack(rcc_tensors),
        )

    # Functions to remove!
    def get_ucc_from_labels_of_batch(self, labels):
        unique_count = torch.unique(labels).size(0)
        unique_count = torch.tensor(unique_count)
        ucc = self.one_hot(unique_count)
        return ucc

    def get_rcc_from_labels_of_batch(self, labels):
        labels = labels.squeeze()
        rcc = torch.zeros(self.num_classes, dtype=torch.int32)
        # Count the occurrences of each class
        for i in range(self.num_classes):
            rcc[i] = (labels == i).sum()
        return rcc

    # util
    def one_hot(self, label):
        # Create a one-hot tensor
        one_hot = torch.zeros(self.num_classes)

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
    # dataset.construct_datasets_with_ucc_and_rcc()
