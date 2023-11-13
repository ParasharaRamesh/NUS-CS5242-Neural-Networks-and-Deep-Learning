import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from params import *
import torch.nn.functional as F


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
            x = nn.LeakyReLU()(x)
            if self.up_sample:
                x = self.upsampling(x)
            out = nn.LeakyReLU()(self.conv1(x))
            out = self.conv2(out)
            if x.shape != out.shape:
                x = self.skip_conv(x)
        else:
            out = nn.LeakyReLU()(self.conv1(x))
            out = nn.LeakyReLU()(self.conv2(out))
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

        # activation functions
        self.leaky = nn.LeakyReLU().to(config.device)
        self.sigmoid = nn.Sigmoid().to(config.device)

        # encoder layers
        self.encoder1_3_8 = nn.Conv2d(3, 8, kernel_size=3, padding=1).to(config.device)  # [b, 8, 32, 32]
        self.encoder2_8_16 = WideResidualBlocks(8, 16, 1).to(config.device)  # [b, 16, 32, 32]
        self.encoder3_16_32 = WideResidualBlocks(16, 32, 1, down_sample=True).to(config.device)  # [b, 32, 16, 16]
        self.encoder4_32_64 = WideResidualBlocks(32, 64, 1, down_sample=True).to(config.device)  # [b, 64, 8, 8]
        self.encoder5_64_128 = WideResidualBlocks(64, 128, 1, down_sample=True).to(config.device)  # [b, 128, 4, 4]
        self.encoder6_128_256 = WideResidualBlocks(128, 256, 1, down_sample=True).to(config.device)  # [b, 256, 2, 2]
        self.encoder7_256_512 = WideResidualBlocks(256, 256, 1, down_sample=True).to(config.device)  # [b, 256, 1, 1]
        self.encoder8_flatten = nn.Flatten().to(config.device)
        self.encoder9_512_700 = nn.Linear(256, 700).to(config.device)
        self.encoder10_700_10 = nn.Linear(700, 10).to(config.device)

        # decoder layers
        self.decoder9_512 = nn.Linear(10, 256).to(config.device)
        self.decoder8_reshape = Reshape(*[256, 1, 1]).to(config.device)
        self.decoder7_512_256 = WideResidualBlocks(256, 256, 1, up_sample=True).to(config.device)  # [b,256, 2, 2]
        self.decoder6_256_128 = WideResidualBlocks(256, 128, 1, up_sample=True).to(config.device)  # [b,128, 4, 4]
        self.decoder5_128_64 = WideResidualBlocks(128, 64, 1, up_sample=True).to(config.device)  # [b, 64, 8, 8]
        self.decoder4_64_32 = WideResidualBlocks(64, 32, 1, up_sample=True).to(config.device)  # [b,32, 16, 16]
        self.decoder3_32_16 = WideResidualBlocks(32, 16, 1, up_sample=True).to(config.device)  # [b,16, 32, 32]
        self.decoder2_16_8 = WideResidualBlocks(16, 8, 1).to(config.device)  # [b,8, 32, 32]
        self.decoder1_8_3 = nn.Conv2d(8, 3, kernel_size=3, padding=1).to(config.device)  # [b, 3, 32, 32]

        print("Autoencoder initialized")

    def forward(self, x):
        #encoder
        e1 = self.leaky(self.encoder1_3_8(x))
        e2 = self.leaky(self.encoder2_8_16(e1))
        e3 = self.leaky(self.encoder3_16_32(e2))
        e4 = self.leaky(self.encoder4_32_64(e3))
        e5 = self.leaky(self.encoder5_64_128(e4))
        e6 = self.leaky(self.encoder6_128_256(e5))
        e7 = self.leaky(self.encoder7_256_512(e6))
        e8 = self.leaky(self.encoder8_flatten(e7))
        e9 = self.leaky(self.encoder9_512_700(e8))
        features = self.sigmoid(self.encoder10_700_10(e9))

        #decoder (making u net like connections for learning better features and ensuring a good reconstruction)
        d9 = self.leaky(self.decoder9_512(features) + e8)
        d8 = self.leaky(self.decoder8_reshape(d9) + e7)
        d7 = self.leaky(self.decoder7_512_256(d8) + e6)
        d6 = self.leaky(self.decoder6_256_128(d7) + e5)
        d5 = self.leaky(self.decoder5_128_64(d6) + e4)
        d4 = self.leaky(self.decoder4_64_32(d5) + e3)
        d3 = self.leaky(self.decoder3_32_16(d4) + e2)
        d2 = self.leaky(self.decoder2_16_8(d3) + e1)
        reconstruction = self.sigmoid(self.decoder1_8_3(d2))

        #return features and reconstructed images
        return features, reconstruction


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
            nn.Linear(110, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 64, dtype=torch.float32),
            nn.Dropout(0.1),
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

        # Stage 3. pass through kde to get output shape (Batch, 128*11)
        kde_prob_distributions = self.kde(features)

        # Stage 4. pass through the ucc_predictor stack to get 4 logits in the end
        ucc_logits = self.ucc_predictor(kde_prob_distributions)

        return ucc_logits, decoded  # (Batch , 4), (Batch * Bag, 3,32,32)

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

        self.shared_predictor_stack = nn.Sequential(
            nn.Linear(110, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(256, 64, dtype=torch.float32),
            nn.Dropout(0.1),
            nn.LeakyReLU()
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

        # Stage 4. pass through common stack
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

    ## Autoencoder model test
    # autoencoder = ResidualAutoencoder().to(device)
    # summary(autoencoder, input_size=(3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # UCC model
    # ucc = UCCModel(device).to(device)
    # summary(ucc, input_size=(config.bag_size, 3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    #         verbose=1)

    # #  RCC model
    # rcc = RCCModel(device).to(device)
    # summary(rcc, input_size=(config.bag_size, 3, 32, 32), device=device, batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    #         verbose=1)
