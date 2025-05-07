import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.data.custom_loader import get_data_loaders


def train():
    device = torch.device("cpu")  # CPU only
    
    # Paths and params
    DATA_DIR = "E:\Projects\Conbias\data-subset1"
    NUM_CLASSES = 5
    EPOCHS = 5
    BATCH_SIZE = 16
    
    # Load Data
    train_loader, val_loader = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # Model
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.3f} - Train Acc: {train_acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "conbias_subset_model.pth")

if __name__ == '__main__':
    train()