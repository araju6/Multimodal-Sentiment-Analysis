import torch
import torch.nn as nn
from torch.nn import functional as F
from SpectrogramAnalysis import SpectrogramCNN
from TextAnalysis import TextLSTM

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class FusionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.CNN = SpectrogramCNN()
        self.LSTM = TextLSTM(input_dim=768)

        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 3),
        )

    def forward(self, x1, x2):
        x1 = self.CNN(x1)
        x2 = self.LSTM(x2)
        new_x = torch.cat([x1, x2], dim= 1)
        out = self.mlp(new_x)
        return out

    