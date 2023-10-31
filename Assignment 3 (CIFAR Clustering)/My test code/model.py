import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from params import *


# KDE layer
class KDE(nn.Module):
    def __init__(self, device=config.device, num_nodes=config.num_nodes, sigma=config.sigma):
        super(KDE, self).__init__()
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.device = device

    def forward(self, data):
        batch_size, bag_size, num_features = data.size()  # Batch, bag, J

        # Create a tensor for the sample points
        k_sample_points = torch.linspace(0, 1, steps=self.num_nodes).repeat(batch_size, bag_size, 1).to(
            self.device)  # B, bag, num_nodes

        # Constants
        k_alfa = 1 / np.sqrt(2 * np.pi * np.square(self.sigma))
        k_beta = -1 / (2 * self.sigma ** 2)

        out_list = []

        for j in range(num_features):
            data_j = data[:, :, j]  # shape (Batch, bag)
            temp_data = data_j.view(-1, bag_size, 1)  # shape (Batch, bag, 1)
            temp_data = temp_data.expand(-1, -1, self.num_nodes)  # shape ( Batch, bag, num_nodes)

            k_diff = k_sample_points - temp_data  # shape ( Batch, bag, num_nodes)
            k_diff_2 = torch.square(k_diff)  # shape ( Batch, bag, num_nodes)
            k_result = k_alfa * torch.exp(k_beta * k_diff_2)  # shape ( Batch, bag, num_nodes)
            k_out_unnormalized = torch.sum(k_result, dim=1)  # (B, num_nodes)
            k_norm_coeff = k_out_unnormalized.sum(dim=1).view(batch_size, 1)  # (B,1)
            k_out = k_out_unnormalized / k_norm_coeff.expand(-1, k_out_unnormalized.size(1))  # (B, num_nodes)

            out_list.append(k_out)
        # out_list is of shape (J, B, num_nodes)
        concat_out = torch.cat(out_list, dim=-1)  # shape is (Batch, J*num_nodes)
        return concat_out  # shape is (Batch, J*num_nodes) -> (1, 8448)


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1, dtype=torch.float),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1, dtype=torch.float),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1, dtype=torch.float),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 16, 48 * 16, dtype=torch.float),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1, dtype=torch.float),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1, dtype=torch.float),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1, dtype=torch.float),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

        # Initialize weights using Xavier initialization with normal distribution
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.to(torch.float)
        encoded = self.encoder(x)
        reshaped_encoded = encoded.view(-1, 48, 4, 4).to(torch.float)
        decoded = self.decoder(reshaped_encoded).to(torch.float)
        return encoded, decoded


# UCC Prediction model
class UCCPredictor(nn.Module):
    def __init__(self, device=config.device, ucc_limit=config.ucc_limit):
        super().__init__()
        # Input size: [Batch, Bag, 48*16]
        # Output size: [Batch, 4]
        self.kde = KDE(device)
        self.stack = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),  # shape 4224
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),  # shape 2112
            nn.ReLU(),
            nn.Linear(2112, 256, dtype=torch.float),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 32, dtype=torch.float),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, ucc_limit, dtype=torch.float),
            nn.Sigmoid()
        )

        # Initialize weights using Xavier initialization with normal distribution
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        kde_prob_distributions = self.kde(x)  # shape (Batch, 8448)
        ucc_logits = self.stack(kde_prob_distributions)  # shape (Batch, 4)
        return ucc_logits


# RCC Prediction model
class RCCPredictor(nn.Module):
    def __init__(self, device=config.device, rcc_limit=config.rcc_limit):
        super().__init__()
        # Input size: [Batch, Bag, 48*16]
        # Output size: [Batch, 4]
        self.kde = KDE(device)
        self.stack = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),  # shape 4224
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),  # shape 2112
            nn.ReLU(),
            nn.Linear(2112, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, rcc_limit),
            nn.ReLU()
        )

        # Initialize weights using Xavier initialization with normal distribution
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        kde_prob_distributions = self.kde(x)  # shape (Batch, 8448)
        rcc_logits = self.stack(kde_prob_distributions)  # shape (Batch, 10)
        return rcc_logits


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autoencoder model test
    # autoencoder = Autoencoder().to(device)
    # summary(autoencoder, input_size=(3, 32, 32), device=device, batch_dim=0,col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # KDE Layer test
    # kde = KDE(device).to(device)
    # summary(kde, input_size=(10, 48 * 16), device=device, batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # UCC layer test
    # ucc_predictor = UCCPredictor(device).to(device)
    # summary(ucc_predictor, input_size=(10, 48 * 16), device=device, batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    # RCC layer test
    # rcc_predictor = RCCPredictor(device).to(device)
    # summary(rcc_predictor, input_size=(10, 48 * 16), device=device, batch_dim=0,
    #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
