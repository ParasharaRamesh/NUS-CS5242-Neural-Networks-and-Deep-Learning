from torchinfo import summary
import torch
import torch.nn as nn


class CIFARClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = self.conv_and_batch_norm_block(in_channels, 64)
        self.conv2 = self.conv_and_batch_norm_block(64, 128, pool=True)
        self.res1 = self.conv_and_batch_norm_block(128, 128)
        self.res2 = self.conv_and_batch_norm_block(128, 128)

        # self.res1 = nn.Sequential(
        #     self.conv_and_batch_norm_block(128, 128),
        #     self.conv_and_batch_norm_block(128, 128),
        # )

        self.conv3 = self.conv_and_batch_norm_block(128, 256, pool=True)
        self.conv4 = self.conv_and_batch_norm_block(256, 512, pool=True)

        self.res3 = self.conv_and_batch_norm_block(512, 512)
        self.res4 = self.conv_and_batch_norm_block(512, 512)

        # self.res2 = nn.Sequential(
        #     self.conv_and_batch_norm_block(512, 512),
        #     self.conv_and_batch_norm_block(512, 512)
        # )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes)
        )

    def conv_and_batch_norm_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        out = self.res1(out1) + out1  # skip connections
        out = out1 + self.res2(out) + out  # multi skip connections
        out = self.conv3(out)
        out2 = self.conv4(out)
        out = self.res3(out2) + out2  # skip connections
        out = out2 + self.res4(out) + out  # multi skip connections
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    model = CIFARClassifier()
    summary(model, (3, 32, 32), batch_dim=0,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
