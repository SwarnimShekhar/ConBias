import os
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights

# ========================
# Configuration
# ========================
TRAIN_DIR = r"E:\Projects\Conbias\data-debiased\train"
VAL_DIR = r"E:\Projects\Conbias\data-debiased\val"
MODEL_SAVE_PATH = "checkpoints/debiased_model.pth"
NUM_CLASSES = 5
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# ========================
# Data Transformations
# ========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========================
# Dataset and Dataloader
# ========================
train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageFolder(root=VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# Model Setup
# ========================
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========================
# Training & Evaluation
# ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"📘 Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        evaluate_model(model, val_loader)

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved at {MODEL_SAVE_PATH}")

def evaluate_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"✅ Validation Accuracy: {acc:.2f}%")

# ========================
# Run Training
# ========================
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer)