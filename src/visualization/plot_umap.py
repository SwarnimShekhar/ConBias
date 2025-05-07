import torch
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def plot_umap(embedding_path, title="UMAP Visualization"):
    # Load data
    data = torch.load(embedding_path)
    embeddings = data["embeddings"].numpy()
    image_paths = data["image_paths"]

    # Extract labels from image folder names
    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                          c=encoded_labels, cmap='Spectral', s=10)
    plt.colorbar(scatter, ticks=np.arange(len(label_encoder.classes_)))
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.show()