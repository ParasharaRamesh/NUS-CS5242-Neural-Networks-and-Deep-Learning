import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split


def create_train_data_loader(batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly adjust brightness, contrast, saturation, and hue
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Randomly crop and resize
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # Random affine transformation
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
        # Randomly erase parts of the image
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

    train_dataset_with_augmentation = torchvision.datasets.CIFAR10(root='./data', download=True, train=True,
                                                                   transform=transform_train)
    return DataLoader(train_dataset_with_augmentation, batch_size=batch_size, shuffle=True)


def create_test_and_validation_data_loader(batch_size=32, validation_split=0.2):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
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
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly adjust brightness, contrast, saturation, and hue
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Randomly crop and resize
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # Random affine transformation
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
        # Randomly erase parts of the image
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
    ])

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
