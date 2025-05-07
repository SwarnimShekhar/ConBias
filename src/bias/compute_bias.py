import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to load embeddings from .pt files
def load_embeddings(embedding_path):
    data = torch.load(embedding_path)
    return data["embeddings"], data["image_paths"]

# Function to compute bias based on cosine similarity
def compute_bias(image_embeddings, concept_embeddings, output_dir):
    """
    Compute co-occurrence bias between image embeddings and concept embeddings.
    Args:
        image_embeddings (tensor): Embeddings for images.
        concept_embeddings (tensor): Embeddings for concepts (e.g., objects or backgrounds).
        output_dir (str): Directory to save bias scores.
    """
    # Calculate cosine similarity between image embeddings and concept embeddings
    image_embeddings = image_embeddings.cpu().numpy()
    concept_embeddings = concept_embeddings.cpu().numpy()

    print(f"Calculating cosine similarity between {image_embeddings.shape[0]} images and {concept_embeddings.shape[0]} concepts...")

    similarity_matrix = cosine_similarity(image_embeddings, concept_embeddings)
    
    # Calculate bias scores for each concept (mean similarity across all images for each concept)
    bias_scores = np.mean(similarity_matrix, axis=0)
    
    # Save bias scores to file
    os.makedirs(output_dir, exist_ok=True)
    bias_file = os.path.join(output_dir, 'bias_scores.npy')
    np.save(bias_file, bias_scores)
    
    print(f"Bias scores saved to {bias_file}")

# Main function to load embeddings and compute bias
def main():
    # Define paths
    image_embeddings_path = "clip_embeddings/train.pt"  # Adjust if needed
    concept_embeddings_path = "clip_embeddings/concepts.pt"  # Adjust if needed
    output_dir = "bias_results"  # Directory to store results
    
    # Load image and concept embeddings
    print("Loading embeddings...")
    image_embeddings, _ = load_embeddings(image_embeddings_path)

    # Load only embeddings from concepts.pt
    concept_data = torch.load(concept_embeddings_path)
    concept_embeddings = concept_data["embeddings"]

    
    # Compute and save bias scores
    print("Computing bias...")
    compute_bias(image_embeddings, concept_embeddings, output_dir)

if __name__ == "__main__":
    main()