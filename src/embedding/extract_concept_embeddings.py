import os
import clip
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_concept_embeddings(concepts, output_path):
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    with torch.no_grad():
        text_tokens = clip.tokenize(concepts).to(device)
        text_embeddings = model.encode_text(text_tokens).cpu()

    torch.save({
        "concepts": concepts,
        "embeddings": text_embeddings
    }, output_path)

    print(f"✅ Saved concept embeddings to {output_path}")

if __name__ == "__main__":
    # You can customize this list as needed
    concepts = [
        "bird", "sky", "grass", "tree", "water", "rock", "branch", "beak", "wing", "feather"
    ]
    output_path = "clip_embeddings/concepts.pt"
    extract_concept_embeddings(concepts, output_path)