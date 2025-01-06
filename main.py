import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

#params
batch_size = 8


