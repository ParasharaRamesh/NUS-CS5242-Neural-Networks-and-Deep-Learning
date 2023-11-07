import torch


class Config:
    drive_path = "/content/drive/MyDrive"
    datasets_path = f"/content/CIFAR-10"
    weights_path = f"{drive_path}/weights"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma = 0.1
    num_nodes = 11

    learning_rate = 0.02 #TODO.5 make it 0.0001 ( 2e-4)
    weight_decay = 1e-4
    grad_clip = 1.5

    batch_size = 1
    ucc_limit = 4
    rcc_limit = 10
    bag_size = 24


config = Config()
