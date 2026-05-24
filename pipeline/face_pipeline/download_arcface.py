#!/usr/bin/env python3
"""Download ArcFace model glintr100.onnx"""
import os
import urllib.request
import sys

weights_dir = "pipeline/face_pipeline/face_embedding_model/weights"
os.makedirs(weights_dir, exist_ok=True)

model_path = os.path.join(weights_dir, "glintr100.onnx")

if os.path.exists(model_path):
    print(f"Model already exists at {model_path}")
    sys.exit(0)

# Try multiple sources
urls = [
    # Source 1: Aliyun OSS (usually fast in Asia)
    "https://insightface-us.oss-us-west-1.aliyuncs.com/files/buffalo_l.zip",
    # Source 2: OneDrive backup
    r"https://onedrive.live.com/download?resid=4A83B6B633B3F5A7!109&authkey=AFZJr7mRcxak0p4",
    # Source 3: Direct from InsightFace GitHub (model zoo)
    "https://github.com/deepinsight/insightface/releases/download/v0.7/glintr100.onnx",
]

for url in urls:
    try:
        print(f"Attempting to download from: {url}")
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Successfully downloaded to {model_path}")
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"  File size: {file_size:.2f} MB")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Failed: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        continue

print("ERROR: Could not download model from any source")
sys.exit(1)
