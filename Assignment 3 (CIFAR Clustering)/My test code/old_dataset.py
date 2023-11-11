from functools import reduce

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from device_data_loader import *
from tqdm import tqdm
import random
from params import *


class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test,
                 debug=False, apply_augmentation=True,
                 batch_size=config.batch_size, bag_size=config.bag_size,
                 ucc_limit=config.ucc_limit, rcc_limit=config.rcc_limit
                 ):
        '''
        Note these are numpy arrays

        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :param x_test:
        :param y_test:
        '''
        self.num_classes = rcc_limit
        self.bag_size = bag_size
        self.ucc_limit = ucc_limit
        self.rcc_limit = rcc_limit
        self.batch_size = batch_size
        self.debug = debug
        self.debug_bag_size = 6
        self.apply_augmentation = apply_augmentation

        # transforms to apply
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

        # converting it all into a tensor (it's not yet one hotified)
        self.x_train = torch.from_numpy(x_train).to(dtype=torch.float32)
        # normalizing the dataset, remove if it doesnt work
        # self.x_train, self.train_mu, self.train_std = self.normalize(self.x_train)
        self.y_train = torch.from_numpy(y_train).to(dtype=torch.float32)

        self.x_test = torch.from_numpy(x_test).to(dtype=torch.float32)
        # normalizing the dataset, remove if it doesnt work
        # self.x_test, self.test_mu, self.test_std = self.normalize(self.x_test)
        self.y_test = torch.from_numpy(y_test).to(dtype=torch.float32)

        # restricting x_val a lot more to 1/10th the test size
        # Generate random indices for sampling without replacement
        random_indices = torch.randperm(len(x_test))
        x_val = x_val[random_indices[:len(x_test)//10]]
        y_val = y_val[random_indices[:len(x_test)//10]]

        self.x_val = torch.from_numpy(x_val).to(dtype=torch.float32)
        # normalizing the dataset, remove if it doesnt work
        # self.x_val, self.val_mu, self.val_std = self.normalize(self.x_val)
        self.y_val = torch.from_numpy(y_val).to(dtype=torch.float32)

        # Dividing all images by 255 to get an image in range 0->1
        self.x_train /= 255
        self.x_test /= 255
        self.x_val /= 255

        print("Converted numpy to torch tensors")

        # create subdatasets ([class_0_imgs, class_1_imgs,... class_9_imgs])
        self.train_sub_datasets = self.create_sub_datasets(self.x_train, self.y_train)
        self.test_sub_datasets = self.create_sub_datasets(self.x_test, self.y_test)
        self.val_sub_datasets = self.create_sub_datasets(self.x_val, self.y_val)

        if not self.debug:
            # create dataloaders
            print("Creating KDE dataloaders")
            self.kde_test_dataloaders = self.create_kde_dataloaders(self.test_sub_datasets)

            print("Created KDE dataloaders, now creating autoencoder dataloaders")
            # batch size is 1 as we care about image level features anyway
            self.autoencoder_test_dataloaders = [DeviceDataLoader(test_sub_dataset, 1) for test_sub_dataset in
                                                 self.test_sub_datasets]
        else:
            # create dataloaders
            print("Creating debug KDE dataloaders")
            self.kde_test_dataloaders = self.create_kde_dataloaders(self.val_sub_datasets)

            print("Created debug KDE dataloaders, now creating debug autoencoder dataloaders")
            # batch size is 1 as we care about image level features anyway
            self.autoencoder_test_dataloaders = [DeviceDataLoader(val_sub_dataset, 1) for val_sub_dataset in
                                                 self.val_sub_datasets]
        print("Created autoencoder dataloaders, now creating ucc dataloaders")
        self.ucc_train_dataloader, self.ucc_test_dataloader, self.ucc_val_dataloader = self.get_dataloaders_for_ucc()
        print("Created ucc dataloaders, now creating rcc dataloaders")
        self.ucc_rcc_train_dataloader, self.ucc_rcc_test_dataloader, self.ucc_rcc_val_dataloader = self.get_dataloaders_for_ucc_and_rcc()

        print("Initilized all dataloaders")

    # create dataloaders
    def create_kde_dataloaders(self, sub_datasets):
        kde_datasets = []

        for chosen_class, pure_sub_dataset in tqdm(enumerate(sub_datasets)):
            total_bags_for_pure_subset = len(pure_sub_dataset) // self.bag_size
            bag_tensors = []

            pure_sub_dataset_idx = 0
            current_bag = self.create_bag()

            while pure_sub_dataset_idx < len(pure_sub_dataset):
                # get the image from this pure sub dataset
                img = pure_sub_dataset[pure_sub_dataset_idx][0]
                bag_idx = pure_sub_dataset_idx % self.bag_size
                current_bag[bag_idx] = img

                if bag_idx == self.bag_size - 1:
                    # the last value has been filled, so add it to the total bags
                    bag_tensors.append(torch.stack(current_bag))

                    # create a new bag for the next set of bags to be filled
                    current_bag = self.create_bag()
                pure_sub_dataset_idx += 1

            kde_datasets.append(TensorDataset(torch.stack(bag_tensors)))

        print("Finished constructing the kde_datasets from the test dataset, now creating dataloaders")

        # NOTE. the batch size here can be different if required.
        kde_data_loaders = [DeviceDataLoader(kde_sub_dataset, self.batch_size) for kde_sub_dataset in kde_datasets]
        return kde_data_loaders

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
            x_class = [torch.tensor(item).permute(2, 0, 1) for item in x[indices]]
            y_class = [torch.tensor(item) for item in y[indices]]

            if len(x_class) > 0 and len(y_class) > 0:
                # Create a TensorDataset for the current class
                class_dataset = TensorDataset(torch.stack(x_class), torch.stack(y_class))

                # Append the current class dataset to the list
                sub_datasets.append(class_dataset)
        assert len(sub_datasets) == self.num_classes
        return sub_datasets

    # pick random image from ith class
    def pick_random_from_ith_sub_dataset(self, sub_datasets, i, is_eval):
        assert 0 <= i < self.num_classes
        sub_dataset = sub_datasets[i]
        sub_dataset_length = len(sub_dataset)
        random_idx = random.randint(0, sub_dataset_length - 1)
        random_img = sub_dataset[random_idx][0]
        if self.apply_augmentation and not is_eval:
            random_transform = random.choice(self.transforms)
            random_img = random_transform(random_img)
        return random_img.to(torch.float32)

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
        loop = self.debug_bag_size if self.debug else total_bags

        # NOTE: we can technically pick more images before I am not enforcing that I am picking every image.
        for b in tqdm(range(loop)):
            # this will keep picking ucc (1 -> 4) in a cyclic manner
            ucc = (b % self.ucc_limit) + 1
            bag_tensor = self.create_bag()

            # you are choosing random classes of size {ucc}. Using this knowledge you have to fill the bag up.
            img_per_class = self.bag_size // ucc
            chosen_classes = random.sample(list(range(self.num_classes)), ucc)
            class_at_each_pos_in_bag = []
            for chosen_class in chosen_classes:
                class_at_each_pos_in_bag.extend([chosen_class] * img_per_class)

            for bag_pos, chosen_class in enumerate(class_at_each_pos_in_bag):
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)

            '''
            #Uncomment this section if you want to try random filling
            random_bag_pos = random.sample(list(range(self.bag_size)), self.bag_size)

            # fill all the values for ucc first and then fill the remaining with random sampling with replacement
            for chosen_class, bag_pos in zip(chosen_classes, random_bag_pos[:len(chosen_classes)]):
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)

            # fill bag_tensor pos by pos
            for bag_pos in random_bag_pos[len(chosen_classes):]:
                chosen_class = random.choice(chosen_classes)
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)

            '''
            bag_tensors.append(torch.stack(bag_tensor))
            #NOTE to self: No need to do one hotification here
            # ucc_tensors.append(self.one_hot(ucc, self.ucc_limit))
            ucc_tensors.append(torch.tensor(ucc-1))

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
        loop = self.debug_bag_size if self.debug else total_bags

        for b in tqdm(range(loop)):  # use this for local testing!
            # for b in tqdm(range(total_bags)):
            # this will keep picking ucc (1 -> 4) in a cyclic manner
            ucc = (b % self.ucc_limit) + 1
            bag_tensor = self.create_bag()
            rcc_tensor = [0] * self.rcc_limit

            # you are choosing random classes of size {ucc}. Using this knowledge you have to fill the bag up.
            img_per_class = self.bag_size // ucc
            chosen_classes = random.sample(list(range(self.num_classes)), ucc)
            class_at_each_pos_in_bag = []
            for chosen_class in chosen_classes:
                class_at_each_pos_in_bag.extend([chosen_class] * img_per_class)

            for bag_pos, chosen_class in enumerate(class_at_each_pos_in_bag):
                bag_tensor[bag_pos] = self.pick_random_from_ith_sub_dataset(sub_datasets, chosen_class, is_eval)
                rcc_tensor[chosen_class] += 1

            '''
            #Uncomment this section if you want to try random filling
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
            '''

            bag_tensors.append(torch.stack(bag_tensor))
            ucc_tensors.append(torch.tensor(ucc-1))
            rcc_tensors.append(torch.tensor(rcc_tensor).to(torch.float32))

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
        return one_hot.to(torch.float32)

    def create_bag(self):
        return [None] * self.bag_size


if __name__ == '__main__':
    splitted_dataset = np.load('../Dataset/splitted_cifar10_dataset.npz')

    x_train = splitted_dataset['x_train']
    y_train = splitted_dataset['y_train']
    x_val = splitted_dataset['x_val']
    y_val = splitted_dataset['y_val']
    x_test = splitted_dataset['x_test']
    y_test = splitted_dataset['y_test']

    dataset = Dataset(x_train, y_train, x_val, y_val, x_test, y_test, debug=False)
    print(f"Len of ucc_train is {len(dataset.ucc_train_dataloader)}")
    print(f"Len of ucc_test is {len(dataset.ucc_test_dataloader)}")
    print(f"Len of ucc_val is {len(dataset.ucc_val_dataloader)}")
