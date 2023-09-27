import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self, lambda_value):
        super(CNNAutoencoder, self).__init__()
        self.lambda_value = lambda_value
        self.encoder = CNNEncoder(lambda_value)
        self.decoder = CNNDecoder(lambda_value)

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstructed_data = self.decoder(latent_features)
        return reconstructed_data


class CNNEncoder(nn.Module):
    def __init__(self, lambda_value):
        super(CNNEncoder, self).__init__()
        self.lambda_value = lambda_value

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x28x28 -> 16x14x14
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # 16x14x14 -> 8x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x14x14 -> 8x7x7
        )
        self.fc1 = nn.Linear(8 * 7 * 7, 392)  # Linear layer with 392 output units
        self.fc2 = nn.Linear(392, lambda_value)  # Linear layer with lambda units

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        latent_features = self.fc2(self.fc1(x))  # Linear layers
        return latent_features


class CNNDecoder(nn.Module):
    def __init__(self, lambda_value):
        super(CNNDecoder, self).__init__()
        self.lambda_value = lambda_value

        # Decoder layers
        self.fc3 = nn.Linear(lambda_value, 392)  # Linear layer with 392 input units
        self.fc4 = nn.Linear(392, 8 * 7 * 7)  # Linear layer to match the flattened shape
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x7x7 -> 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x14x14 -> 1x28x28
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc4(self.fc3(x))  # Linear layers
        x = x.view(x.size(0), 8, 7, 7)  # Reshape to match the decoder input
        reconstructed_image = self.decoder(x)
        return reconstructed_image
