# Multi-Modal Image Retrieval System

This project implements a **multi-modal image retrieval system** using **CLIP** for embedding extraction and **L2 distance** for searching the most relevant images based on a text query.

## Features
- Extracts **image embeddings** using OpenAI's CLIP model.
- Stores **test image embeddings** for efficient retrieval.
- Uses **L2 distance** to find the most similar images to a given text query.
- Provides a **FastAPI backend** for querying.
- A **Next.js frontend** to interact with the system.

---

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/your-repo/multi-modal-image-retrieval.git
cd multi-modal-image-retrieval
```

###  Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download & Prepare Dataset
Ensure that your test images are stored in:
```
multi-modal-image-retrieval/data/raw/test_data_v2
```
If your dataset is in a different location, update the `config_settings.py` file.

---

## Running the System

### Extract Image Embeddings
```bash
python src/feature_extractor.py
```
This generates `test_embeddings.npy` in `multi-modal-image-retrieval/data`.

### Start the Backend API (FastAPI)
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### Start the Frontend (Next.js)
```bash
cd frontend
npm install  # Only required for the first time
npm run dev
```
The frontend will be available at: **[http://localhost:3000](http://localhost:3000)**.

---

## API Usage

### Search for Images using Text Query
#### **Endpoint:** `POST /search`
#### **Request Body:**
```json
{
  "text": "cat",
  "top_k": 5
}
```
#### **Response:**
```json
{
  "query": "cat",
  "results": [
    {
      "image": "http://127.0.0.1:8000/data/raw/test_data_v2/image1.jpg",
      "distance": 1.21
    },
    {
      "image": "http://127.0.0.1:8000/data/raw/test_data_v2/image2.jpg",
      "distance": 1.22
    }
  ]
}
```

---

## Configuration
Modify `config_settings.py` to update paths:
```python
# Paths
DATA_DIR = "/absolute/path/to/multi-modal-image-retrieval/data"
TEST_FOLDER = os.path.join(DATA_DIR, "raw/test_data_v2")
TEST_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "test_embeddings.npy")
```

---

## Troubleshooting
### Backend Not Starting?
- Check if the embeddings exist:
  ```bash
  ls multi-modal-image-retrieval/data/test_embeddings.npy
  ```
  If missing, **run the feature extractor**:  
  ```bash
  python src/feature_extractor.py
  ```

- Check if FastAPI is running:
  ```bash
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
  ```

### Frontend Not Displaying Images?
- Check the API response in the browser at `http://127.0.0.1:8000/search`.
- Ensure CORS is enabled in `api.py`.
- Check if images exist in `multi-modal-image-retrieval/data/raw/test_data_v2`.

---

**Developed by:** _[Sirshen_  

