import torch
import torch.nn as nn
from torchinfo import summary
class CNNAutoencoder(nn.Module):
    def __init__(self, lambda_value):
        super().__init__()
        self.lambda_value = lambda_value
        self.encoder = CNNEncoder(lambda_value)
        self.decoder = CNNDecoder(lambda_value)

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstructed_data = self.decoder(latent_features)
        return reconstructed_data

class CNNEncoder(nn.Module):
    def __init__(self, lambda_value):
        super().__init__()
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
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 7 * 7, 392),
            nn.ReLU()
        )  # Linear layer with 392 output units
        self.fc2 = nn.Sequential(
            nn.Linear(392, lambda_value),
            nn.ReLU()
        )  # Linear layer with lambda units

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        latent_features = self.fc2(x)  # Linear layers
        return latent_features


class CNNDecoder(nn.Module):
    def __init__(self, lambda_value):
        super(CNNDecoder, self).__init__()
        self.lambda_value = lambda_value

        # Define the decoder layers using nn.Sequential
        self.decoder = nn.Sequential(
            # Linear layer: Lambda -> 392
            nn.Linear(self.lambda_value, 392),
            nn.ReLU(),

            # Reshape to 8x7x7
            nn.Unflatten(1, (8, 7, 7)),

            # ConvTranspose layers
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),  # 8x7x7 -> 8x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=1, padding=1),  # 8x14x14 -> 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),  # 16x14x14 -> 16x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # 16x28x28 -> 1x28x28
            # nn.ReLU()  # Apply relu
        )

    def forward(self, x):
        return self.decoder(x)


# if __name__ == '__main__':
#     encoder = CNNEncoder(2)
#     summary(encoder, (1,28,28), batch_dim = 0, col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose = 1)
#     decoder = CNNDecoder(2)
#     summary(decoder, (2,), batch_dim = 0, col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose = 1)

