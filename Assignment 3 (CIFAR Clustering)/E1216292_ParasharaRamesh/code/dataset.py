import numpy as np
import torch
from torch.utils.data import TensorDataset
from device_data_loader import *


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
        self.train_loader = DeviceDataLoader(self.train_dataset, 1)
        self.val_loader = DeviceDataLoader(self.val_dataset, 1)
        self.test_loader = DeviceDataLoader(self.test_dataset, 1)

    #get UCC
    def construct_datasets_with_ucc(self):
        pass

    #get both UCC and RCC
    def construct_datasets_with_ucc_and_rcc(self):
        pass

    #util
    def one_hot(self, labels):
        # Create an empty one-hot tensor
        one_hot_tensor = torch.zeros((labels.size(0), self.num_classes), dtype=torch.float32)

        # Use scatter to fill in the one-hot tensor
        one_hot_tensor.scatter_(1, labels.view(-1, 1), 1)

        return one_hot_tensor


if __name__ == '__main__':
    splitted_dataset = np.load('splitted_cifar10_dataset.npz')

    x_train = splitted_dataset['x_train']
    print(f"x_train shape :{x_train.shape}")

    y_train = splitted_dataset['y_train']
    print(f"y_train shape :{y_train.shape}")

    x_val = splitted_dataset['x_val']
    print(f"x_val shape :{x_val.shape}")

    y_val = splitted_dataset['y_val']
    print(f"y_val shape :{y_val.shape}")

    x_test = splitted_dataset['x_test']
    print(f"x_test shape :{x_test.shape}")

    y_test = splitted_dataset['y_test']
    print(f"y_test shape: {y_test.shape}")

    dataset = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)
