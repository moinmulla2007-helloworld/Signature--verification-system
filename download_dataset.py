import os
import zipfile
import urllib.request
import shutil

# Dataset URL from open GitHub repository (CEDAR sample subset)
DATASET_URL = "https://github.com/Aayush-K/Signature-Verification/archive/refs/heads/master.zip"
ZIP_PATH = "dataset_temp.zip"
EXTRACT_DIR = "dataset_temp"

print("Downloading real signature dataset...")
urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

print("Extracting files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# Paths in extracted repo
base_extracted = os.path.join(EXTRACT_DIR, "Signature-Verification-master", "data")
real_src = os.path.join(base_extracted, "real")
forged_src = os.path.join(base_extracted, "forged")

# Destination paths
genuine_dst = os.path.join("dataset", "genuine")
forged_dst = os.path.join("dataset", "forged")

os.makedirs(genuine_dst, exist_ok=True)
os.makedirs(forged_dst, exist_ok=True)

# Copy genuine signatures
if os.path.exists(real_src):
    for f in os.listdir(real_src):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(os.path.join(real_src, f), genuine_dst)

# Copy forged signatures
if os.path.exists(forged_src):
    for f in os.listdir(forged_src):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(os.path.join(forged_src, f), forged_dst)

# Cleanup temp files
os.remove(ZIP_PATH)
shutil.rmtree(EXTRACT_DIR)

print("Dataset successfully extracted and placed into dataset/genuine and dataset/forged!")