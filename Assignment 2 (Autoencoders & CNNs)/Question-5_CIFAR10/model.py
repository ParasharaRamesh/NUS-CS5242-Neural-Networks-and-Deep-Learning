from torchinfo import summary
import torch
import torch.nn as nn


class CIFARClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, num_classes),  # Adjust the input size based on your image dimensions
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # converting to softmax
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = CIFARClassifier()
    summary(model,(3,32,32) ,batch_dim=0, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1)
