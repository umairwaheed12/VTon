#!/bin/bash
set -e

# RunPod Setup Script for FVTON (Virtual Try-On)
# This script automates cloning, dependencies, and launching.

echo "==========================================="
echo "   FVTON RUNPOD SETUP & LAUNCH SCRIPT      "
echo "==========================================="
echo "TIP: To keep this running in the background, run this script inside a 'tmux' session."
echo ""

# 0. System Dependencies (including tmux)
echo "Step 0: Checking system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    apt-get update && apt-get install -y tmux
fi

# 1. Environment Check
echo "Step 0: Pre-flight check..."
if [ -f "models_downloader.py" ]; then
    echo "   Already in project root."
else
    if [ -d "VTon" ] && [ -f "VTon/models_downloader.py" ]; then
        echo "   VTon directory exists, entering..."
        cd VTon
    else
        echo "   Cloning VTon repository..."
        git clone https://github.com/umairwaheed12/VTon.git
        cd VTon
    fi
fi

# 2. Run the environment setup and model download
echo "Step 1: Setting up environment and downloading models..."
python3 models_downloader.py

# 3. Fix ONNX Runtime GPU (CUDNN) library discovery
echo "Step 2: Configuring NVIDIA library paths for GPU acceleration..."
# Find all library directories in the 'nvidia' python package
NVIDIA_LIBS=$(python3 -c "import os, nvidia; nvidia_path = os.path.dirname(nvidia.__file__); 
lib_paths = []
for root, dirs, files in os.walk(nvidia_path):
    if 'lib' in dirs:
        lib_paths.append(os.path.join(root, 'lib'))
print(':'.join(lib_paths))" 2>/dev/null)

if [ ! -z "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
    echo "   ✅ Added NVIDIA libs to LD_LIBRARY_PATH"
    echo "   ℹ ONNX Runtime 1.19.2+ will now find CUDNN 9 libs automatically."
fi

# 4. Launch Fooocus
echo "Step 3: Launching Fooocus..."
# --listen: allows external connections
# --port 7865: matches RunPod default exposed port
# --share: creates a gradio.live link
python3 Fooocus/launch.py --listen --port 7865 --share
