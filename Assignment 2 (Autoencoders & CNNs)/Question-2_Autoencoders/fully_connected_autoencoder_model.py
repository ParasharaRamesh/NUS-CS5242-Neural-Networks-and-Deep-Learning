import torch
import torch.nn as nn


# Define your Fully Connected Autoencoder
class FC_Autoencoder(nn.Module):
    def __init__(self, lambda_value):
        super(FC_Autoencoder, self).__init__()
        self.lambda_value = lambda_value
        self.encoder = FullyConnectedEncoder(lambda_value)
        self.decoder = FullyConnectedDecoder(lambda_value)

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstructed_data = self.decoder(latent_features)
        return reconstructed_data
class FullyConnectedEncoder(nn.Module):
    def __init__(self, Lambda):
        super(FullyConnectedEncoder, self).__init__()
        self.Lambda = Lambda

        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 196),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(196, 49),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(49, Lambda),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder forward pass
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.fc2(x)
        latent_features = self.fc3(x)  # Extract latent features here
        return latent_features


class FullyConnectedDecoder(nn.Module):
    def __init__(self, Lambda):
        super(FullyConnectedDecoder, self).__init__()
        self.Lambda = Lambda

        self.fc4 = nn.Sequential(
            nn.Linear(Lambda, 49),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(49, 196),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(196, 28 * 28),
            nn.ReLU()
        )

    def forward(self, x):
        # Decoder forward pass
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = x.view(x.size(0), 1, 28, 28)  # Reshape to 28x28x1
        return x
