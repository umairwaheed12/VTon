#!/bin/bash

# RunPod Setup Script for Fooocus Virtual Try-On
echo "Starting Fooocus RunPod Setup..."

# Update and install system dependencies if needed (optional)
# apt-get update && apt-get install -y libgl1

# Install core requirements
echo "Installing Fooocus requirements..."
pip install -r requirements_versions.txt

# Install specific dependencies for Virtual Try-On
echo "Installing Virtual Try-On dependencies (Mediapipe, ONNX-GPU)..."
pip install mediapipe onnxruntime-gpu

# Optional: Download specific models for dress.py and masking.py
echo "Downloading small models for Virtual Try-On..."
python modules/download_small_models.py

# Launch Fooocus
# --listen: allows external connections
# --port 7865: matches RunPod default exposed port
# --share: creates a gradio.live link as a backup
echo "Launching Fooocus..."
python launch.py --listen --port 7865 --share
