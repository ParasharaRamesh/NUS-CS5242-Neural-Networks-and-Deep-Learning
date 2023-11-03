import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from params import *
import torch.nn.functional as F

# Fancy approach which didnt work
'''

class ResidualZeroPaddingBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            first_block=False,
            down_sample=False,
            up_sample=False,
    ):
        super(ResidualZeroPaddingBlock, self).__init__()
        self.first_block = first_block
        self.down_sample = down_sample
        self.up_sample = up_sample

        if self.up_sample:
            self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self.down_sample else 1,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2 if self.down_sample else 1,
        )

        # Initialize the weights and biases
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.xavier_uniform_(self.skip_conv.weight)

    def forward(self, x):
        if self.first_block:
            x = nn.ReLU()(x)
            if self.up_sample:
                x = self.upsampling(x)
            out = nn.ReLU()(self.conv1(x))
            out = self.conv2(out)
            if x.shape != out.shape:
                x = self.skip_conv(x)
        else:
            out = nn.ReLU()(self.conv1(x))
            out = nn.ReLU()(self.conv2(out))
        return x + out


class WideResidualBlocks(nn.Module):
    def __init__(
            self, in_channels, out_channels, n, down_sample=False, up_sample=False
    ):
        super(WideResidualBlocks, self).__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualZeroPaddingBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    first_block=(i == 0),
                    down_sample=down_sample,
                    up_sample=up_sample,
                )
                for i in range(n)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class NewAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
            ),
            WideResidualBlocks(
                16,
                32,
                1
            ),
            WideResidualBlocks(
                32,
                64,
                1,
                down_sample=True
            ),
            WideResidualBlocks(
                64,
                128,
                1,
                down_sample=True,
            ),  # [b,128,8,8]
            WideResidualBlocks(
                128,
                256,
                1,
                down_sample=True,
            ),  # [b,256,4,4] -> 4096
            WideResidualBlocks(
                256,
                256,
                1,
                down_sample=True,
            ),  # [b,256,2,2] -> 1024
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            WideResidualBlocks(
                256,
                256,
                1,
                up_sample=True,
            ),
            WideResidualBlocks(
                256,
                128,
                1,
                up_sample=True,
            ),
            WideResidualBlocks(
                128,
                64,
                1,
                up_sample=True,
            ),
            WideResidualBlocks(
                64,
                32,
                1,
                up_sample=True,
            ),
            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid()
        )

        print("Autoencoder initialized")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


'''


class Reshape(nn.Module):
    def __init__(self, *target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(x.size(0), *self.target_shape)


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, dtype=torch.float32),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, dtype=torch.float32),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, dtype=torch.float32),  # [batch, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Reshape(*[128 * 4 * 4]),
            nn.Linear(128 * 4 * 4, 128 * 4 * 4),
            nn.Dropout(0.1),
            Reshape(*[128, 4, 4]),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, dtype=torch.float32),  # [batch, 32, 8, 8]
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, dtype=torch.float32),  # [batch, 16, 16, 16]
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, dtype=torch.float32),  # [batch, 3, 32, 32]
            nn.Sigmoid()
        )

        # Initialize weights using Xavier initialization with normal distribution
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        print("Autoencoder model initialized!")

    def forward(self, x):
        x = x.to(torch.float32)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded).to(torch.float32)
        return encoded, decoded


# KDE layer
class KDE(nn.Module):
    def __init__(self, device=config.device, num_nodes=config.num_nodes, sigma=config.sigma):
        super(KDE, self).__init__()
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.device = device
        print("KDE Layer initialized")

    def forward(self, data):
        batch_size, num_instances, num_features = data.shape

        # Create sample points
        k_sample_points = (
            torch.linspace(0, 1, steps=config.num_nodes)
            .repeat(batch_size, num_instances, 1)
            .to(device)
        )

        # Calculate constants
        k_alpha = 1 / np.sqrt(2 * np.pi * config.sigma ** 2)
        k_beta = -1 / (2 * config.sigma ** 2)

        # Iterate over features and calculate kernel density estimation for each feature
        out_list = []
        for i in range(num_features):
            one_feature = data[:, :, i: i + 1].repeat(1, 1, config.num_nodes)
            k_diff_2 = (k_sample_points - one_feature) ** 2
            k_result = k_alpha * torch.exp(k_beta * k_diff_2)
            k_out_unnormalized = k_result.sum(dim=1)
            k_norm_coeff = k_out_unnormalized.sum(dim=1).view(-1, 1)
            k_out = k_out_unnormalized / k_norm_coeff.repeat(
                1, k_out_unnormalized.size(1)
            )
            out_list.append(k_out)

        # Concatenate the results
        concat_out = torch.cat(out_list, dim=-1).to(self.device)
        return concat_out


# UCC Prediction model
class UCCPredictor(nn.Module):
    def __init__(self, device=config.device, ucc_limit=config.ucc_limit):
        super().__init__()
        # Input size: [Batch, Bag, 128*4*4]
        # Output size: [Batch, 4]
        self.kde = KDE(device)

        # input (Batch, 2048*11)
        self.stack = nn.Sequential(
            Reshape(*[11, 2048]),
            nn.Conv1d(in_channels=11, out_channels=11, kernel_size=2, stride=2),  # output shape (Batch, 11, 1024)
            nn.BatchNorm1d(11),
            nn.ReLU(),
            nn.Conv1d(in_channels=11, out_channels=11, kernel_size=2, stride=2),  # output shape (Batch, 11, 512)
            nn.BatchNorm1d(11),
            nn.ReLU(),
            Reshape(*[11 * 512]),
            nn.Linear(5632, 512, dtype=torch.float32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 128, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 32, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, ucc_limit, dtype=torch.float32)
        )

        # Input size: [Batch, Bag, 128*4*4]
        self.stack_without_kde = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 256, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 32, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, ucc_limit, dtype=torch.float32)
        )

        # Initialize weights using Xavier initialization with normal distribution
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        print("UCC Predictor model initialized")

    def forward(self, x):
        # Commenting out KDE as it is not learning much

        kde_prob_distributions = self.kde(x)  # shape (Batch, 22528)
        ucc_logits = self.stack(kde_prob_distributions)  # shape (Batch, 4)

        # This is without KDE at all
        # batch_size, bag_size, features = x.size()
        # x = x.view(-1, bag_size * features).to(config.device)
        # x = nn.MaxPool1d(4)(x)
        # x = nn.AvgPool1d(4)(x)
        # x = nn.Linear(x.size(1), 1024, device=config.device)(x)
        # ucc_logits = self.stack_without_kde(x)
        return ucc_logits


# Combined UCC model
class CombinedUCCModel(nn.Module):
    def __init__(self, device=config.device, autoencoder_model=None):
        super().__init__()
        if autoencoder_model:
            self.autoencoder = autoencoder_model
        else:
            self.autoencoder = Autoencoder()

        self.ucc_predictor = UCCPredictor(device)
        print("Combined UCC model initialized")

    def forward(self, batch):
        # Input size: [batch, bag, 3, 32, 32]
        # output size: [batch, 4] (ucc_logits), [batch * bag,3,32,32] ( decoded images)

        # Stage 1. pass through autoencoder
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        encoded, decoded = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)

        # Stage 2. use the autoencoder latent features to pass through the ucc predictor
        # encoded shape is now (Batch* Bag, 128,4,4) -> (Batch, Bag, 128*4*4)
        encoded = encoded.view(batch_size, bag_size, encoded.size(1) * encoded.size(2) * encoded.size(3))
        ucc_logits = self.ucc_predictor(encoded)

        return ucc_logits, decoded


# RCC Prediction model
class RCCPredictor(nn.Module):
    def __init__(self, device=config.device, rcc_limit=config.rcc_limit):
        super().__init__()
        # Input size: [Batch, Bag, 1024]
        # Output size: [Batch, 4]
        self.kde = KDE(device)
        self.stack = nn.Sequential(
            Reshape(*[11, 2048]),
            nn.Conv1d(in_channels=11, out_channels=11, kernel_size=2, stride=2),  # output shape (Batch, 11, 1024)
            nn.BatchNorm1d(11),
            nn.ReLU(),
            nn.Conv1d(in_channels=11, out_channels=11, kernel_size=2, stride=2),  # output shape (Batch, 11, 512)
            nn.BatchNorm1d(11),
            nn.ReLU(),
            Reshape(*[11 * 512]),
            nn.Linear(5632, 512, dtype=torch.float32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 128, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 32, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, rcc_limit, dtype=torch.float32)
        )

        # Input size: [Batch, Bag, 128*4*4]
        self.stack_without_kde = nn.Sequential(
            nn.Linear(6144, 1024, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 256, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 32, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, rcc_limit, dtype=torch.float32),
            nn.ReLU()
        )

        # Initialize weights using Xavier initialization with normal distribution
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        print("RCC Predictor module initilized")

    def forward(self, x):
        # Uncomment this to try it with KDE
        kde_prob_distributions = self.kde(x)  # shape (Batch, 8448)
        rcc_logits = self.stack(kde_prob_distributions)  # shape (Batch, 10)

        '''
        #This is without KDE
        x = x.view(config.batch_size * config.bag_size, 128, 4, 4).to(config.device)
        x = nn.Conv2d(128, 64, 4, stride=2, padding=1, dtype=torch.float32, device=config.device)(x)
        x = x.view(-1)
        rcc_logits = self.stack_without_kde(x)
        '''
        return rcc_logits


# Combined RCC model
class CombinedRCCModel(nn.Module):
    def __init__(self, device=config.device, autoencoder_model=None):
        super().__init__()

        if autoencoder_model:
            self.autoencoder = autoencoder_model
        else:
            self.autoencoder = Autoencoder()

        self.ucc_predictor = UCCPredictor(device)
        self.rcc_predictor = RCCPredictor(device)

        print("Combined RCC Predictor initialized")

    def forward(self, batch):
        # Input size: [batch, bag, 3, 32, 32]
        # output size: [batch, 4] (ucc_logits), [batch, 10] (rcc_logits), [batch * bag,3,32,32] ( decoded images)

        # Stage 1. pass through autoencoder
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        encoded, decoded = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)
        # encoded shape is now (Batch* Bag, 256,4,4) -> (Batch, Bag, 256*4*4)
        encoded = encoded.view(batch_size, bag_size, encoded.size(1) * encoded.size(2) * encoded.size(3))

        # Stage 2. use the autoencoder latent features to pass through the ucc predictor
        ucc_logits = self.ucc_predictor(encoded)

        # Stage 3. use the autoencoder latent features to pass through the rcc predictor
        rcc_logits = self.rcc_predictor(encoded)
        return rcc_logits, ucc_logits, decoded


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autoencoder model test
    # autoencoder = NewAutoencoder().to(device)
    # autoencoder = Autoencoder().to(device)
    # summary(autoencoder, input_size=(3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # KDE Layer test
    # kde = KDE(device).to(device)
    # summary(kde, input_size=(10, 48 * 16), device=device, batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # Combined UCC model
    combined_ucc = CombinedUCCModel(device).to(device)
    summary(combined_ucc, input_size=(config.bag_size, 3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            verbose=1)

    # Combined RCC model
    combined_rcc = CombinedRCCModel(device).to(device)
    summary(combined_rcc, input_size=(config.bag_size, 3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            verbose=1)
