import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np

#KDE layer
class KernelDensityEstimator(nn.Module):
    def __init__(self, device, num_nodes=11, sigma=0.1, num_features=10):
        super(KernelDensityEstimator, self).__init__()
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.num_features = num_features
        self.device = device

    '''
    TODO.x check if the KDE code is correct by checking with the original implementation!
    '''
    def forward(self, data):
        batch_size, num_samples, num_features = data.size()

        # Create a tensor for the sample points
        k_sample_points = torch.linspace(0, 1, steps=self.num_nodes).repeat(batch_size, num_samples, 1).to(self.device)

        # Constants
        k_alfa = 1 / np.sqrt(2 * np.pi * np.square(self.sigma))
        k_beta = -1 / (2 * self.sigma ** 2)

        out_list = []
        for i in range(self.num_features):
            temp_data = data[:, :, i].view(batch_size, num_samples, 1)

            k_diff = k_sample_points - temp_data.expand(-1, -1, self.num_nodes)
            k_diff_2 = torch.square(k_diff)
            k_result = k_alfa * torch.exp(k_beta * k_diff_2)
            k_out_unnormalized = torch.sum(k_result, dim=1)
            k_norm_coeff = k_out_unnormalized.sum(dim=1).view(batch_size, 1)
            k_out = k_out_unnormalized / k_norm_coeff.expand(-1, k_out_unnormalized.size(1))
            out_list.append(k_out)
        concat_out = torch.cat(out_list, dim=-1)
        return concat_out #return shape of (batch, 10 * 11)


#Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48*16, 48*16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reshaped_encoded = encoded.view(-1, 48, 4, 4)
        decoded = self.decoder(reshaped_encoded)
        return encoded, decoded



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autoencoder model test
    # autoencoder = Autoencoder().to(device)
    # summary(autoencoder, input_size=(3, 32, 32), device=device, batch_dim=0,col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)

    #KDE Layer test
    kde = KernelDensityEstimator(device).to(device)
    summary(kde, input_size=(10, 48*16), device=device, batch_dim=0,col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
