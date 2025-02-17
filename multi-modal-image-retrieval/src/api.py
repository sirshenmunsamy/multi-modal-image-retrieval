from fastapi import FastAPI
import faiss
import torch
import clip
import numpy as np
from pydantic import BaseModel
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"

# Ensure `src/` is in Python's path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Paths
FAISS_INDEX_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/faiss_index.bin"
EMBEDDINGS_FILE = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data/embeddings.npy"

# Initialize FastAPI
app = FastAPI()

# Load CLIP Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", DEVICE)

# Load FAISS index
faiss_index = faiss.read_index(FAISS_INDEX_FILE)

# Load image paths
data = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
image_paths = data["paths"]

# Define a request model
class QueryRequest(BaseModel):
    text: str
    top_k: int = 5  # Default to top 5 results

def get_text_embedding(text):
    """Convert a text query into an embedding using CLIP."""
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(DEVICE)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()
    return text_embedding

@app.post("/search")
async def search_images(query: QueryRequest):
    """Search FAISS for the most relevant images."""
    text_embedding = get_text_embedding(query.text)
    
    distances, indices = faiss_index.search(text_embedding, query.top_k)

    results = [{"image": image_paths[idx], "distance": float(distances[0][i])} for i, idx in enumerate(indices[0])]

    return {"query": query.text, "results": results}

@app.get("/")
async def root():
    """API Root Endpoint."""
    return {"message": "Multi-Modal Image Retrieval API is running!"}

