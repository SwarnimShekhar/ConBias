import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import nn

# Define transformations (make sure this matches the training transforms)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the debiased model
model = models.resnet50(weights=None)  # Use no weights here since we're loading the checkpoint
num_classes = 5  # Make sure this matches your dataset's number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the model weights
model.load_state_dict(torch.load("checkpoints/debiased_model.pth"))
model.eval()

# Generate bias scores on the test set
test_dataset = ImageFolder(root='E:\\Projects\\Conbias\\data-debiased\\val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

bias_scores = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = torch.max(outputs, 1)[1]
        
        # Calculate bias score (difference between predicted and actual labels)
        bias_score = (predicted != labels).cpu().numpy()
        bias_scores.extend(bias_score)

# Convert bias_scores to a numpy array and ensure it's in integer format
bias_scores = np.array(bias_scores, dtype=int)

# Plot the Bias Score Histogram
plt.figure(figsize=(8, 6))
plt.hist(bias_scores, bins=2, edgecolor='black')
plt.title('Bias Score Histogram on Debiased Test Set')
plt.xlabel('Bias (0 = No Bias, 1 = Bias)')
plt.ylabel('Frequency')

# Save the histogram plot
plt.savefig("bias_score_histogram.png")
print("Bias Score Histogram saved as bias_score_histogram.png")

# Calculate per-concept bias scores
concepts = test_dataset.classes
concept_bias = {concept: 0 for concept in concepts}

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Count the bias occurrences for each concept
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                concept_bias[concepts[labels[i]]] += 1

# Normalize per-concept bias scores
total_samples = len(test_dataset)
concept_bias = {k: v / total_samples for k, v in concept_bias.items()}

# Plot the Per-Concept Bias Barplot
concepts_sorted = sorted(concept_bias, key=concept_bias.get, reverse=True)
bias_values = [concept_bias[concept] for concept in concepts_sorted]

plt.figure(figsize=(10, 8))
plt.barh(concepts_sorted, bias_values, color='salmon')
plt.title('Per-Concept Bias Scores on Debiased Test Set')
plt.xlabel('Bias Score')
plt.ylabel('Concept')

# Save the barplot
plt.savefig("per_concept_bias_barplot.png")
print("Per-Concept Bias Barplot saved as per_concept_bias_barplot.png")