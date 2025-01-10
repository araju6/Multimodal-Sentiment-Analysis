import torch
import torch.nn as nn
from torch.nn import functional as F

from wav2img import wavImgConverter
from wav2text import wavTextConverter
from FusionModel import FusionModel
import pandas as pd


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

#params
batch_size = 3

data = pd.read_csv('test_dataset.csv').sample(frac=1, random_state=69)

train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

model = FusionModel().to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

wav2img = wavImgConverter()
wav2text = wavTextConverter()

for epoch in range(10):
    model.train()
    train_loss = 0
    
    for i in range(0, len(train_data), batch_size):
        batch_files = train_data.iloc[i:i+batch_size]['file_path'].tolist()
        labels = torch.tensor(train_data.iloc[i:i+batch_size]['category'].tolist()).to(device)
        
        images = wav2img.convert(batch_files)
        texts = wav2text.convert(batch_files)
        
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_files = test_data.iloc[i:i+batch_size]['file_path'].tolist()
            labels = torch.tensor(test_data.iloc[i:i+batch_size]['category'].tolist()).to(device)
            
            images = wav2img.convert(batch_files)
            texts = wav2text.convert(batch_files)
            
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_train_loss = train_loss / (len(train_data) // batch_size)
    avg_test_loss = test_loss / (len(test_data) // batch_size)
    test_acc = 100 * correct / total
    
    print(f'Epoch {epoch+1}:')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    print('-' * 50)