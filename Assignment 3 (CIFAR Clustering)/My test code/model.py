import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from params import *
import torch.nn.functional as F

'''
Old approach

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
            nn.Sigmoid() #TODO.1 dont use sigmoid! ( input to KDE is sigmoid!)
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


class Reshape(nn.Module):
    def __init__(self, *target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(x.size(0), *self.target_shape)


class ResidualAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=3,
                padding=1,
            ),
            WideResidualBlocks(
                8,
                16,
                1
            ),
            WideResidualBlocks(
                16,
                32,
                1,
                down_sample=True
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
            ),  # [b,128,4,4]
            WideResidualBlocks(
                128,
                256,
                1,
                down_sample=True,
            ),  # [b,256,2,2] -> 1024
            WideResidualBlocks(
                256,
                512,
                1,
                down_sample=True,
            ),  # [b,512,1,1] -> 512
            nn.Sigmoid()
        )

        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 679),
            nn.LeakyReLU(),
            nn.Linear(679, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10), #Input to kde is just 10
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            WideResidualBlocks(
                512,
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
            WideResidualBlocks(
                32,
                16,
                1,
                up_sample=True,
            ),
            WideResidualBlocks(
                16,
                8,
                1
            ),
            nn.Conv2d(
                8,
                3,
                kernel_size=3,
                padding=1,
            )
        )

        print("Autoencoder initialized")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        features = self.feature_extractor(encoded)
        return features, decoded


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
            .to(self.device)
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

# UCC model
class UCCModel(nn.Module):
    def __init__(self, device=config.device, autoencoder_model=None, ucc_limit=config.ucc_limit):
        super().__init__()
        if autoencoder_model:
            self.autoencoder = autoencoder_model
        else:
            self.autoencoder = ResidualAutoencoder()

        self.kde = KDE(device)
        self.ucc_predictor = nn.Sequential(
            nn.Linear(110, 384),
            nn.LeakyReLU(),
            nn.Linear(384, 192, dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(192, 64, dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(64, ucc_limit, dtype=torch.float32)
        )

        print("UCC model initialized")

    def forward(self, batch):
        # Input size: [batch, bag, 3, 32, 32]
        # output size: [batch, 4] (ucc_logits), [batch * bag,3,32,32] ( decoded images)

        # Stage 1. pass through autoencoder
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        features, decoded = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)

        # Stage 2. use the autoencoder latent features to pass through the ucc predictor
        # features shape is now (Batch* Bag, 128) -> (Batch, Bag, 128)
        features = features.view(batch_size, bag_size, features.size(1))

        #Stage 3. pass through kde to get output shape (Batch, 128*11)
        kde_prob_distributions = self.kde(features)

        # Stage 4. pass through the ucc_predictor stack to get 4 logits in the end
        ucc_logits = self.ucc_predictor(kde_prob_distributions)

        return ucc_logits, decoded # (Batch , 4), (Batch * Bag, 3,32,32)

    def get_encoder_features(self, batch):
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        features, _ = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)
        return features

    def get_kde_distributions(self, batch):
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        features, _ = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)
        features = features.view(batch_size, bag_size, features.size(1))

        kde_prob_distributions = self.kde(features)
        return kde_prob_distributions


# RCC model
class RCCModel(nn.Module):
    def __init__(self, device=config.device, autoencoder_model=None, ucc_limit=config.ucc_limit, rcc_limit=config.rcc_limit):
        super().__init__()
        if autoencoder_model:
            self.autoencoder = autoencoder_model
        else:
            self.autoencoder = ResidualAutoencoder()

        self.kde = KDE(device)

        self.shared_predictor_stack =  nn.Sequential(
            nn.Linear(110, 384),
            nn.LeakyReLU(),
            nn.Linear(384, 192, dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(192, 64, dtype=torch.float32),
            nn.LeakyReLU(),
        )

        self.ucc_predictor = nn.Linear(64, ucc_limit, dtype=torch.float32)
        self.rcc_predictor = nn.Linear(64, rcc_limit, dtype=torch.float32)

        print(" UCC model initialized")

    def forward(self, batch):
        # Input size: [batch, bag, 3, 32, 32]
        # output size: [batch, 4] (ucc_logits), [batch * bag,3,32,32] ( decoded images)

        # Stage 1. pass through autoencoder
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        features, decoded = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)

        # Stage 2. use the autoencoder latent features to pass through the ucc predictor
        # features shape is now (Batch* Bag, 128) -> (Batch, Bag, 128)
        features = features.view(batch_size, bag_size, features.size(1))

        # Stage 3. pass through kde to get output shape (Batch, 128*11)
        kde_prob_distributions = self.kde(features)

        #Stage 4. pass through common stack
        common_features = self.shared_predictor_stack(kde_prob_distributions)

        # Stage 5. get the ucc logits
        ucc_logits = self.ucc_predictor(common_features)

        # Stage 6. get the rcc logits
        rcc_logits = self.rcc_predictor(common_features)

        return rcc_logits, ucc_logits, decoded  # , (Batch , 10), (Batch , 4), (Batch * Bag, 3,32,32)

    def get_encoder_features(self, batch):
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        features, _ = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)
        return features

    def get_kde_distributions(self, batch):
        batch_size, bag_size, num_channels, height, width = batch.size()
        batches_of_image_bags = batch.view(batch_size * bag_size, num_channels, height, width).to(torch.float32)
        features, _ = self.autoencoder(
            batches_of_image_bags
        )  # we are feeding in Batch*bag images of shape (3,32,32)
        features = features.view(batch_size, bag_size, features.size(1))

        kde_prob_distributions = self.kde(features)
        return kde_prob_distributions


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autoencoder model test
    # autoencoder = ResidualAutoencoder().to(device)
    # summary(autoencoder, input_size=(3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # UCC model
    ucc = UCCModel(device).to(device)
    summary(ucc, input_size=(config.bag_size, 3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            verbose=1)

    #  RCC model
    rcc = RCCModel(device).to(device)
    summary(rcc, input_size=(config.bag_size, 3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            verbose=1)
