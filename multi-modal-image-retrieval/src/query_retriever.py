import faiss
import numpy as np
import torch
import clip
import os

os.environ["OMP_NUM_THREADS"] = "1"

# Paths
FAISS_INDEX_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/faiss_index.bin"
TEST_EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/test_embeddings.npy"

# Load CLIP Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", DEVICE)

def load_test_embeddings():
    """Load **test embeddings** instead of searching FAISS."""
    data = np.load(TEST_EMBEDDINGS_FILE, allow_pickle=True).item()
    return data["embeddings"], data["paths"]

def get_text_embedding(text):
    """Convert a text query into an embedding using CLIP."""
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(DEVICE)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()
    return text_embedding

def compute_nearest_neighbors(text_query, top_k=10):
    """Find nearest test images to the query embedding."""
    test_embeddings, test_image_paths = load_test_embeddings()  # ✅ Load test embeddings

    # Convert query to an embedding
    text_embedding = get_text_embedding(text_query)

    # Compute L2 distance between text embedding and test embeddings
    distances = np.linalg.norm(test_embeddings - text_embedding, axis=1)

    # Get top-K closest images
    top_k_indices = np.argsort(distances)[:top_k]

    results = [
        {
            "image": f"http://127.0.0.1:8000/data/raw/test_data_v2/{os.path.basename(test_image_paths[idx])}",
            "distance": float(distances[idx])
        }
        for idx in top_k_indices
    ]

    return results

if __name__ == "__main__":
    query = input("Enter a text query: ")
    results = compute_nearest_neighbors(query)

    print("\n🔍 **Top Matching Images:**")
    for result in results:
        print(f"📸 Image: {result['image']} | 🔢 Distance: {result['distance']}")
