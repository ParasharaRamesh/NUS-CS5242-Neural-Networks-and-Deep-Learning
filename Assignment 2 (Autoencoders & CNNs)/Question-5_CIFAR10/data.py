import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset


def create_train_data_loader(batch_size=32):
    # normal dataset
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normal_transforms = [
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(*stats, inplace=True)  # Normalize pixel values
    ]
    train_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True,
                                                 transform=transforms.Compose(normal_transforms))

    # augmenting dataset
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transforms_to_apply = [
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(7),  # Randomly rotate images by up to 7 degrees
        # Randomly adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(*stats, inplace=True)  # Normalize pixel values
    ]
    train_dataset_with_augmentation = torchvision.datasets.CIFAR10(root='./data', download=True, train=True,
                                                                   transform=transforms.Compose(transforms_to_apply))

    # Combine the datasets using ConcatDataset
    combined_dataset = ConcatDataset([train_dataset, train_dataset_with_augmentation])

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)


def create_test_and_validation_data_loader(batch_size=32, validation_split=0.15):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)  # Normalize pixel values
    ])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False,
                                                transform=transform_test)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate the number of samples to use for validation
    num_total_samples = len(test_dataset_loader.dataset)
    num_validation_samples = int(num_total_samples * validation_split)
    num_test_samples = num_total_samples - num_validation_samples

    # Split the test dataset into test and validation sets
    validation_dataset, test_dataset = random_split(test_dataset_loader.dataset,
                                                    [num_validation_samples, num_test_samples])

    # Create DataLoaders for validation and test sets
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return test_loader, validation_loader


def create_train_data_loader_with_num_instances(num_instances, batch_size=32):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transforms_to_apply = [
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(7),  # Randomly rotate images by up to 7 degrees
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(*stats, inplace=True)  # Normalize pixel values
    ]
    transform_train = transforms.Compose(transforms_to_apply)

    # Load the full CIFAR-10 training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True,
                                                      transform=transform_train)

    # Create a subset of the training dataset based on the number of instances per class
    class_indices = [torch.where(torch.tensor(full_train_dataset.targets) == class_label)[0] for class_label in
                     range(10)]

    sampled_indices = []
    for indices in class_indices:
        sampled_indices.extend(random.sample(indices.tolist(), num_instances))

    train_dataset_subset = Subset(full_train_dataset, sampled_indices)

    # Create a DataLoader for the subset
    train_loader = DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True)

    return train_loader


# if __name__ == '__main__':
#     train_data_loader = create_train_data_loader(32)
#     for batch_idx, data in enumerate(train_data_loader):
#         images, labels = data
#         break
#     print("Success")
