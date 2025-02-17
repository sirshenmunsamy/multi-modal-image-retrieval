import torch
import clip
from PIL import Image
import os
import numpy as np

# Load CLIP model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", DEVICE)

# Define dataset folders
IMAGE_FOLDERS = ["/Users/sirshenmunsamy/Desktop/SB Case Study/data/raw/test_data_v2",
                 "/Users/sirshenmunsamy/Desktop/SB Case Study/data/raw/train_data"]

OUTPUT_FILE = "data/embeddings.npy"

def extract_image_embeddings(image_folders, output_file):
    """ Extracts CLIP embeddings for all images in the dataset. """
    image_embeddings = []
    image_paths = []

    for folder in image_folders:
        print(f"📂 Processing images in {folder}...")

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    embedding = model.encode_image(image).cpu().numpy()
                image_embeddings.append(embedding)
                image_paths.append(img_path)

            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")

    # Convert to NumPy and save
    if image_embeddings:
        image_embeddings = np.vstack(image_embeddings)
        np.save(output_file, {"embeddings": image_embeddings, "paths": image_paths})
        print(f"✅ Saved {len(image_embeddings)} image embeddings to {output_file}")
    else:
        print("⚠ No images processed. Check dataset paths.")

if __name__ == "__main__":
    extract_image_embeddings(IMAGE_FOLDERS, OUTPUT_FILE)