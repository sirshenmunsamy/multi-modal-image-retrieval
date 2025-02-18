import torch
import clip
from PIL import Image
import os
import numpy as np

# Load CLIP model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", DEVICE)

# Define dataset folders
TRAIN_FOLDER = "/Users/sirshenmunsamy/Desktop/SB Case Study/data/raw/train_data"
TEST_FOLDER = "/Users/sirshenmunsamy/Desktop/SB Case Study/data/raw/test_data_v2"

TRAIN_EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/train_embeddings.npy"
TEST_EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/test_embeddings.npy"

def extract_embeddings(image_folder, output_file):
    """Extract CLIP embeddings for all images in the dataset."""
    image_embeddings = []
    image_paths = []

    print(f"📂 Processing images in {image_folder}...")

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)

        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy()
            image_embeddings.append(embedding)
            image_paths.append(img_path)

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

    if image_embeddings:
        image_embeddings = np.vstack(image_embeddings)
        np.save(output_file, {"embeddings": image_embeddings, "paths": image_paths})
        print(f"✅ Saved {len(image_embeddings)} image embeddings to {output_file}")
    else:
        print("⚠ No images processed. Check dataset paths.")

if __name__ == "__main__":
    extract_embeddings(TRAIN_FOLDER, TRAIN_EMBEDDINGS_FILE)
    extract_embeddings(TEST_FOLDER, TEST_EMBEDDINGS_FILE)