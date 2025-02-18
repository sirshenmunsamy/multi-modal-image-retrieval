import faiss
import numpy as np

# File paths
TRAIN_EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/train_embeddings.npy"
TEST_EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/test_embeddings.npy"
FAISS_INDEX_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/faiss_index.bin"

def load_embeddings():
    """Load **train** embeddings for FAISS indexing."""
    data = np.load(TRAIN_EMBEDDINGS_FILE, allow_pickle=True).item()
    return data["embeddings"], data["paths"]

def build_faiss_index(embeddings):
    """Create a FAISS index for fast nearest-neighbor search."""
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(embeddings)  # Add embeddings to FAISS index
    return index

def save_faiss_index(index, file_path):
    """Save FAISS index to disk."""
    faiss.write_index(index, file_path)
    print(f"✅ FAISS index saved to {file_path}")

if __name__ == "__main__":
    print("📥 Loading train embeddings...")
    embeddings, train_image_paths = load_embeddings()

    print("⚡ Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)

    print("💾 Saving FAISS index...")
    save_faiss_index(faiss_index, FAISS_INDEX_FILE)

    print("✅ FAISS index created and saved successfully!")