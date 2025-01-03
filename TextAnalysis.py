import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class TextLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(input_size= 20, hidden_size=20, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(20, 3), nn.Dropout(0.5))


    def forward(self, x):
        out, _ = self.LSTM(x)
        out = self.fc(out[:, -1, :])
        return out

    