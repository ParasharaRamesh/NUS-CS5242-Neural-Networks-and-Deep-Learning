import numpy as np
import torch
from torch.utils.data import TensorDataset
from device_data_loader import *
from tqdm import tqdm


class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=16):
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
        self.batch_size = batch_size

        # converting it all into a tensor (its not yet one hotified)
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_val = torch.from_numpy(x_val)
        self.y_val = torch.from_numpy(y_val)
        self.x_test = torch.from_numpy(x_test)
        self.y_test = torch.from_numpy(y_test)

        # create datasets
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.val_dataset = TensorDataset(self.x_val, self.y_val)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)

        # create loaders for going through the dataset to create the new final dataset
        self.train_loader = DeviceDataLoader(self.train_dataset, batch_size=self.batch_size)
        self.val_loader = DeviceDataLoader(self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DeviceDataLoader(self.test_dataset, batch_size=self.batch_size)

    # get UCC
    def construct_datasets_with_ucc(self):
        train_dataset_with_ucc = self.construct_dataset_with_ucc(self.train_loader)
        val_dataset_with_ucc = self.construct_dataset_with_ucc(self.val_loader)
        test_dataset_with_ucc = self.construct_dataset_with_ucc(self.test_loader)

        return train_dataset_with_ucc, val_dataset_with_ucc, test_dataset_with_ucc

    def construct_dataset_with_ucc(self, dataloader):
        image_tensors = []
        ucc_tensors = []

        for data in tqdm(dataloader):
            images, labels = data

            ucc = self.get_ucc_from_labels_of_batch(labels)

            image_tensors.append(images)
            ucc_tensors.append(ucc)


        return TensorDataset(
            torch.stack(image_tensors),
            torch.stack(ucc_tensors)
        )

    def construct_datasets_with_ucc_and_rcc(self):
        train_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.train_loader)
        val_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.val_loader)
        test_dataset_with_ucc_and_rcc = self.construct_dataset_with_ucc_and_rcc(self.test_loader)

        return train_dataset_with_ucc_and_rcc, val_dataset_with_ucc_and_rcc, test_dataset_with_ucc_and_rcc

    # get both UCC and RCC
    def construct_dataset_with_ucc_and_rcc(self, dataloader):
        image_tensors = []
        ucc_tensors = []
        rcc_tensors = []

        for data in tqdm(dataloader):
            images, labels = data

            #get ucc
            ucc = self.get_ucc_from_labels_of_batch(labels)

            #get rcc
            rcc = self.get_rcc_from_labels_of_batch(labels)

            image_tensors.append(images)
            ucc_tensors.append(ucc)
            rcc_tensors.append(rcc)


        return TensorDataset(
            torch.stack(image_tensors),
            torch.stack(ucc_tensors),
            torch.stack(rcc_tensors),
        )

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
        one_hot[label-1] = 1
        return one_hot


if __name__ == '__main__':
    splitted_dataset = np.load('splitted_cifar10_dataset.npz')

    x_train = splitted_dataset['x_train']
    y_train = splitted_dataset['y_train']
    x_val = splitted_dataset['x_val']
    y_val = splitted_dataset['y_val']
    x_test = splitted_dataset['x_test']
    y_test = splitted_dataset['y_test']

    dataset = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)
    dataset.construct_datasets_with_ucc_and_rcc()
