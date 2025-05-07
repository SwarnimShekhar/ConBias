import os
import random
import shutil
from PIL import Image

def debias_images(train_dir, output_dir, debiasing_factor=0.7):
    """
    Apply debiasing to the dataset by sampling a subset of images from biased classes.
    
    Arguments:
    - train_dir: Path to the training directory.
    - output_dir: Path to the output directory where debiased images will be saved.
    - debiasing_factor: The percentage of images to retain for each class (default is 70%).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each class folder in the training directory
    for class_folder in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_folder)
        
        if os.path.isdir(class_path):
            print(f"Processing class folder: {class_folder}")  # Debug print
            
            # Create a corresponding folder in the output directory
            output_class_path = os.path.join(output_dir, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Found {len(image_files)} images in class {class_folder}")  # Debug print
            
            # Apply debiasing (randomly sample based on the debiasing factor)
            num_images_to_keep = int(len(image_files) * debiasing_factor)
            sampled_images = random.sample(image_files, num_images_to_keep)
            
            print(f"Keeping {num_images_to_keep} images out of {len(image_files)}")  # Debug print
            
            # Copy the selected images to the output directory
            for img_file in sampled_images:
                img_path = os.path.join(class_path, img_file)
                output_img_path = os.path.join(output_class_path, img_file)
                
                try:
                    img = Image.open(img_path)
                    img.save(output_img_path)
                    print(f"Saved {img_file} to {output_class_path}")  # Debug print
                except Exception as e:
                    print(f"Error saving image {img_file}: {e}")

    print(f"✅ Dataset debiasing complete. Saved to {output_dir}")

if __name__ == "__main__":
    # Define the paths for training and output directories
    train_dir = "E:/Projects/Conbias/data-subset1/train"
    output_dir = "E:/Projects/Conbias/data-debiased"
    
    # Run the debiasing function
    debias_images(train_dir, output_dir)