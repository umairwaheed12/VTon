"""
Download OpenPose ControlNet model for SDXL to preserve person pose during virtual try-on.
"""
import os
import requests
from tqdm import tqdm

# Configuration
MODEL_URL = "https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "Fooocus", "models", "controlnet")
MODEL_FILENAME = "controlnet_openpose_sdxl.safetensors"

def download_file(url, destination):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Stream download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=MODEL_FILENAME,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"\n✅ Download complete: {destination}")
    print(f"File size: {os.path.getsize(destination) / (1024**3):.2f} GB")

def main():
    destination = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    # Check if file already exists
    if os.path.exists(destination):
        print(f"⚠️  Model already exists at: {destination}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    try:
        download_file(MODEL_URL, destination)
        print("\n🎉 OpenPose ControlNet model is ready to use!")
        print(f"Location: {destination}")
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nAlternative: Download manually from:")
        print(MODEL_URL)
        print(f"And place it in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
