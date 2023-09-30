from torchinfo import summary
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class CIFARClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.15):
        super(CIFARClassifier, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512)
        )

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = CIFARClassifier()
    summary(model,(3,32,32) ,batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
