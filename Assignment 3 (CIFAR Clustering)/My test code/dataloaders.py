import numpy as np
from params import *
from ucc_dataset import *
from rcc_dataset import *
from cifar_dataset import *
from device_data_loader import *
class Dataloaders:
    def __init__(self):
        data = np.load(config.datasets_path)
        x_train, y_train = data["x_train"], data["y_train"]
        x_val, y_val = data["x_val"], data["y_val"]
        x_test, y_test = data["x_test"], data["y_test"]

        #construct ucc datasets
        self.ucc_train_dataset = UCCDataset(x_train, y_train, config.train_steps, True)
        self.ucc_val_dataset = UCCDataset(x_val, y_val, config.val_steps, False)
        self.ucc_test_dataset = UCCDataset(x_test, y_test, config.test_steps, False)

        #construct rcc datasets
        self.rcc_train_dataset = RCCDataset(x_train, y_train, config.train_steps, True)
        self.rcc_val_dataset = RCCDataset(x_val, y_val, config.val_steps, False)
        self.rcc_test_dataset = RCCDataset(x_test, y_test, config.test_steps, False)

        #construct cifar datasets
        self.cifar_train_dataset = CIFARDataset(x_train, y_train, config.train_steps, True)
        self.cifar_val_dataset = CIFARDataset(x_val, y_val, config.val_steps, False)
        self.cifar_test_dataset = CIFARDataset(x_test, y_test, config.test_steps, False)

        #delete the extra unnecessary space
        del data
        del x_train
        del x_test
        del x_val
        del y_train
        del y_test
        del y_val
    def get_ucc_dataloaders(self):
        train_loader = DeviceDataLoader(self.ucc_train_dataset, config.batch_size)
        val_loader = DeviceDataLoader(self.ucc_val_dataset, config.batch_size)
        test_loader = DeviceDataLoader(self.ucc_test_dataset, config.batch_size)
        return train_loader, val_loader, test_loader

    def get_rcc_dataloaders(self):
        train_loader = DeviceDataLoader(self.rcc_train_dataset, config.batch_size)
        val_loader = DeviceDataLoader(self.rcc_val_dataset, config.batch_size)
        test_loader = DeviceDataLoader(self.rcc_test_dataset, config.batch_size)
        return train_loader, val_loader, test_loader

    def get_cifar_dataloaders(self):
        train_loader = DeviceDataLoader(self.cifar_train_dataset, config.batch_size)
        val_loader = DeviceDataLoader(self.cifar_val_dataset, config.batch_size)
        test_loader = DeviceDataLoader(self.cifar_test_dataset, config.batch_size)
        return train_loader, val_loader, test_loader

if __name__ == '__main__':
    dataloaders = Dataloaders()
    dataloaders.get_ucc_dataloaders()
    dataloaders.get_rcc_dataloaders()
    dataloaders.get_cifar_dataloaders()
    print("done")