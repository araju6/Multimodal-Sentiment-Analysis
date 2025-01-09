import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class TextLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=20, hidden_size=20, num_layers=2, batch_first=True)
        self.attention = nn.Linear(20, 1)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(20, 3))

    def forward(self, x):
        out, _ = self.LSTM(x)

        attention_weights = F.softmax(self.attention(out), dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)

        out = self.fc(context_vector)
        return out


    