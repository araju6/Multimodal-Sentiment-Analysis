import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class TextLSTM(nn.Module):
    
    def __init__(self, input):
        super().__init__()
        self.string = input
    


