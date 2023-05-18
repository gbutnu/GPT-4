import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import GPT2Model
from datasets import GPT2Dataset

def train(model, train_dataset, val_dataset, batch_size, lr, num_epochs, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            model.train()

        end_time = time.time()
        print('Epoch {}/{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Time: {:.2f}s'.format(
            epoch+1, num_epochs, train_loss, val_loss, end_time-start_time))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2Model().to(device)
    train_dataset = GPT2Dataset('data/train.txt')
    val_dataset = GPT2Dataset('data/val.txt')
    batch_size = 32
    lr = 0.001
    num_epochs = 10
    train(model, train_dataset, val_dataset, batch_size, lr, num_epochs, device)