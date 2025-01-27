import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class TextLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        out, _ = self.LSTM(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


    