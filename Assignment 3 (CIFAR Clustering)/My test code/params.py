import torch


class Config:
    drive_path = "/content/drive/MyDrive"
    datasets_path = f"/content/CIFAR-10"
    weights_path = f"{drive_path}/weights"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 0.01
    weight_decay = 1e-4
    grad_clip = 0.1

    batch_size = 2
    ucc_limit = 4
    rcc_limit = 10
    bag_size = 10


config = Config()
