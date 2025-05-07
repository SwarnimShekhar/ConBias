import os
import clip
import torch
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_clip_embeddings(data_dir, output_dir):
    model, preprocess = clip.load("ViT-B/32", device=device)
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue

        embeddings = []
        image_paths = []

        # Go inside each class folder
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = model.encode_image(image)
                    embeddings.append(embedding.cpu())
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")

        if embeddings:
            embeddings_tensor = torch.cat(embeddings, dim=0)
            save_path = os.path.join(output_dir, f"{split}.pt")
            torch.save({
                "embeddings": embeddings_tensor,
                "image_paths": image_paths
            }, save_path)
            print(f"✅ Saved {split} embeddings to {save_path}")

    print("🎯 Feature extraction complete.")

if __name__ == "__main__":
    data_dir = "data-subset1"         # Contains train/ and val/ (with bird folders inside)
    output_dir = "clip_embeddings"    # Will save train.pt and val.pt
    extract_clip_embeddings(data_dir, output_dir)
