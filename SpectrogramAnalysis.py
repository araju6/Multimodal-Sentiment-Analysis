import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class SpectrogramCNN(nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(1,1),),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1),),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 48, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, image, targets = None):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



