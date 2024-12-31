import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class SpectrogramCNN(nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = nn.Sequential(
            nn.Conv2d()
        )

    def forward(self, image, targets = None):
        if targets == None:


