import faiss
import numpy as np
import torch
import clip
from PIL import Image
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Paths
FAISS_INDEX_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/faiss_index.bin"
EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/embeddings.npy"

# Load CLIP Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", DEVICE)

def load_faiss_index():
    """Load the FAISS index from file."""
    return faiss.read_index(FAISS_INDEX_FILE)

def load_image_paths():
    """Load image paths from the saved embeddings file."""
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    return data["paths"]

def get_text_embedding(text):
    """Convert a text query into an embedding using CLIP."""
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(DEVICE)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()
    return text_embedding

def search_faiss(text_query, top_k=5):
    """Search the FAISS index for the top-K most similar images to the text query."""
    faiss_index = load_faiss_index()
    image_paths = load_image_paths()
    
    # Convert the query into an embedding
    text_embedding = get_text_embedding(text_query)
    
    # Perform the search
    distances, indices = faiss_index.search(text_embedding, top_k)
    
    # Retrieve the matching image paths
    results = [(image_paths[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    
    return results

if __name__ == "__main__":
    # Example query
    query = input("Enter a text query: ")
    results = search_faiss(query)

    print("\n🔍 **Top Matching Images:**")
    for img_path, score in results:
        print(f"📸 Image: {img_path} | 🔢 Distance: {score}")