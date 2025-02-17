import kagglehub
import shutil
import os

# Download dataset to the default location
dataset_path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")

# Define the correct project folder
PROJECT_DATA_PATH = "data/raw"

# Ensure the destination folder exists
os.makedirs(PROJECT_DATA_PATH, exist_ok=True)

# Move dataset from KaggleHub default location to project folder
for item in os.listdir(dataset_path):
    source_path = os.path.join(dataset_path, item)
    destination_path = os.path.join(PROJECT_DATA_PATH, item)

    if os.path.isdir(source_path):
        shutil.move(source_path, destination_path)
    else:
        shutil.copy2(source_path, destination_path)

print("✅ Dataset successfully moved to:", PROJECT_DATA_PATH)