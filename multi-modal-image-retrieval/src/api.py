from fastapi import FastAPI
import torch
import clip
import numpy as np
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_settings import TEST_FOLDER, TEST_EMBEDDINGS_FILE, DEVICE  # ✅ Fixed Import

# ✅ Ensure the data directory exists before mounting
DATA_DIR = os.path.dirname(TEST_FOLDER)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"✅ Created missing directory: {DATA_DIR}")

# ✅ Load CLIP Model
model, preprocess = clip.load("ViT-B/32", DEVICE)

# ✅ Load test embeddings safely
def load_test_embeddings():
    """Load test image embeddings & paths."""
    if not os.path.exists(TEST_EMBEDDINGS_FILE):
        raise FileNotFoundError(f"❌ Error: Test embeddings file not found at {TEST_EMBEDDINGS_FILE}")

    data = np.load(TEST_EMBEDDINGS_FILE, allow_pickle=True).item()
    return data["embeddings"], data["paths"]

# ✅ Compute nearest neighbors
def compute_nearest_neighbors(text_query, top_k=10):
    """Find nearest test images to the query embedding."""
    test_embeddings, test_image_paths = load_test_embeddings()

    # Convert query to an embedding
    with torch.no_grad():
        text_tokens = clip.tokenize([text_query]).to(DEVICE)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()

    # ✅ Normalize embeddings
    text_embedding = text_embedding / np.linalg.norm(text_embedding)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    # Compute L2 distance
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

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static files (images)
app.mount("/data/raw/test_data_v2", StaticFiles(directory=TEST_FOLDER, check_dir=True), name="test_data_v2")
# ✅ Define input model
class QueryRequest(BaseModel):
    text: str
    top_k: int = 10  # Default top_k value

@app.post("/search")
async def search_images(query: QueryRequest):
    """Use L2 distance to retrieve test images."""
    try:
        results = compute_nearest_neighbors(query.text, query.top_k)
        return {"query": query.text, "results": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Multi-Modal Image Retrieval API is running!"}