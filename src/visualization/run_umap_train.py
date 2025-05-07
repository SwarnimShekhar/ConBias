from plot_umap import plot_umap

if __name__ == "__main__":
    embedding_path = "clip_embeddings/train.pt"
    plot_umap(embedding_path, title="UMAP - CLIP Features (Train Set)")