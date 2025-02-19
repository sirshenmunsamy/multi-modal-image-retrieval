import os
import torch

# ✅ **Base Directory**
PROJECT_ROOT = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval"

# ✅ **Dataset Directory (ONLY TEST DATA)**
DATA_DIR = "/Users/sirshenmunsamy/Desktop/SB Case Study/multi-modal-image-retrieval/data"
TEST_FOLDER = os.path.join(DATA_DIR, "raw/test_data_v2")  # This is the only folder we need

# ✅ **Embeddings Directory**
TEST_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "test_embeddings.npy")

# ✅ **Device Selection**
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ **Ensure Required Directories Exist**
for folder in [TEST_FOLDER, DATA_DIR]:
    if not os.path.exists(folder):
        print(f"❌ Error: Directory {folder} does not exist! Please check paths.")
    else:
        print(f"✅ Directory exists: {folder}")