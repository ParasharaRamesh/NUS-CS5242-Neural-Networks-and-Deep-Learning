import torch


class Config:
    # for local
    datasets_path = f"../Dataset/splitted_cifar10_dataset.npz"
    weights_path = f"../weights"

    # drive_path = "/content/drive/MyDrive"
    # datasets_path = f"/content/CIFAR-10"
    # weights_path = f"{drive_path}/weights"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigma = 0.1
    num_nodes = 11

    learning_rate = 1e-4
    weight_decay = 1e-4
    grad_clip = 0.7

    # batch_size = 20
    batch_size = 2
    ucc_limit = 4
    rcc_limit = 10
    bag_size = 24
    num_classes = 10

    # train_steps = 100000
    train_steps = 100 # for local testing
    test_steps = 1000
    val_steps = 10
    # val_steps = 100
    # debug_steps = 1000
    debug_steps = 10
    saver_steps = 10


config = Config()
