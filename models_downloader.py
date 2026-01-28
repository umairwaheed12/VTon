import os
import shutil
from pathlib import Path
import sys
import subprocess
import platform

def install_system_dependencies():
    """Install system dependencies (Linux/RunPod only)."""
    if platform.system() == "Linux":
        print("⏳ Checking/Installing system dependencies (apt-get)...")
        try:
            # Common CV libraries + ffmpeg + wget
            subprocess.check_call("apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg wget", shell=True)
            print("✅ System dependencies verified.")
        except Exception as e:
            print(f"⚠ Warning: Failed to install system dependencies: {e}")
            print("  (This is expected if you don't have root/sudo access)")

def install_python_dependencies():
    """Install required Python packages."""
    print("⏳ Installing Python dependencies...")
    
    # Superset of requirements from all scripts
    deps = [
        # Core
        "huggingface_hub",
        "gradio",
        "numpy<2.3.0",  # Pinned for compatibility
        "opencv-python-headless", # Better for server environments
        
        # ML / Torch
        "torch>=2.4.0",
        "torchvision", 
        "accelerate",
        "einops", 
        "timm", 
        "ultralytics",
        
        # Inference Engines
        "onnxruntime-gpu", # Assuming RunPod has GPU
        
        # CV / Processing
        "mediapipe==0.10.9",
        "protobuf<3.20.4", # Often needed for MediaPipe compatibility
        "transformers==4.39.0", # STRICTLY PINNED for Moondream2 compatibility
        "pillow",
        "segment-anything"
    ]
    
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)
    print("✅ Python dependencies installed.")

# Run installations first
install_system_dependencies()
install_python_dependencies()

from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    """
    Download all models required for gradio_app.py, masking.py, and dresss.py.
    """
    base_dir = Path(__file__).parent.absolute()
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Downloading models to: {models_dir}")

    # ------------------------------------------------------------------
    # 1. Moondream2 (from vikhyatk/moondream2)
    # ------------------------------------------------------------------
    print("\n[1/4] Downloading Moondream2...")
    moondream_path = models_dir / "moondream2"
    if not moondream_path.exists():
        snapshot_download(
            repo_id="vikhyatk/moondream2",
            local_dir=moondream_path,
            local_dir_use_symlinks=False,
            revision="2024-04-02" # Pinned revision
        )
        print(f"✓ Downloaded to: {moondream_path}")
    else:
         print(f"✓ Already exists: {moondream_path}")


    # ------------------------------------------------------------------
    # 2. SegFormer B3 Fashion (from sayeed99/segformer-b3-fashion)
    # ------------------------------------------------------------------
    print("\n[2/4] Downloading SegFormer B3 Fashion...")
    segformer_b3_path = models_dir / "segformer-b3-fashion"
    if not segformer_b3_path.exists():
        snapshot_download(
            repo_id="sayeed99/segformer-b3-fashion",
            local_dir=segformer_b3_path,
            local_dir_use_symlinks=False  # Clean copy for portability
        )
        print(f"✓ Downloaded to: {segformer_b3_path}")
    else:
        print(f"✓ Already exists: {segformer_b3_path}")

    # ------------------------------------------------------------------
    # 3. LIP Parsing Model (from pngwn/IDM-VTON Space)
    # ------------------------------------------------------------------
    print("\n[3/4] Downloading LIP Parsing Model (ONNX)...")
    humanparsing_dir = models_dir / "humanparsing"
    humanparsing_dir.mkdir(exist_ok=True)
    target_path = humanparsing_dir / "parsing_lip.onnx"
    
    if not target_path.exists():
        try:
            print("   Attempting download from HuggingFace Space (pngwn/IDM-VTON)...")
            hf_hub_download(
                repo_id="pngwn/IDM-VTON",
                repo_type="space",
                filename="ckpt/humanparsing/parsing_lip.onnx",
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            
            # Move from downloaded structure to simple structure
            # Downloaded: models/ckpt/humanparsing/parsing_lip.onnx
            source_path = models_dir / "ckpt" / "humanparsing" / "parsing_lip.onnx"
            
            if source_path.exists():
                shutil.move(str(source_path), str(target_path))
                print(f"   Moved to: {target_path}")
                # Cleanup
                if (models_dir / "ckpt").exists():
                    shutil.rmtree(models_dir / "ckpt")
            else:
                # Maybe it downloaded directly if flat? No, usually follows path.
                pass
                
        except Exception as e:
            print(f"⚠ Hub download failed: {e}")
            print("   Falling back to direct URL download...")
            try:
                # Direct resolve URL for the file in the space
                url = "https://huggingface.co/spaces/pngwn/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"
                subprocess.check_call(["wget", "-O", str(target_path), url])
            except Exception as e2:
                print(f"❌ Failed to download LIP model: {e2}")
    else:
        print(f"✓ Already exists: {target_path}")

    print(f"✓ LIP Model verified at: {target_path}")

    # ------------------------------------------------------------------
    # 4. SegFormer B2 Clothes ONNX (from mattmdjaga/segformer_b2_clothes)
    # ------------------------------------------------------------------
    print("\n[4/4] Downloading SegFormer B2 Clothes (ONNX)...")
    b2_clothes_dir = models_dir / "SegFormerB2Clothes"
    b2_clothes_dir.mkdir(exist_ok=True)
    target_b2 = b2_clothes_dir / "segformer_b2_clothes.onnx"

    if not target_b2.exists():
        # The file in the repo is 'onnx/model.onnx'
        hf_hub_download(
            repo_id="mattmdjaga/segformer_b2_clothes",
            filename="onnx/model.onnx",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        source_b2 = models_dir / "onnx" / "model.onnx"
        
        if source_b2.exists():
            shutil.move(str(source_b2), str(target_b2))
            try:
                shutil.rmtree(models_dir / "onnx")
            except:
                pass
    else:
        print(f"✓ Already exists: {target_b2}")
            
    print(f"✓ Downloaded to: {target_b2}")
    
    # ------------------------------------------------------------------
    # 5. SAM Model (from facebook/sam-vit-base)
    # ------------------------------------------------------------------
    print("\n[5/5] Downloading SAM (Segment Anything) Model...")
    sam_dir = models_dir / "sam"
    sam_dir.mkdir(exist_ok=True)
    target_sam = sam_dir / "sam_vit_b_01ec64.pth"
    
    if not target_sam.exists():
        try:
            # Download from facebook/sam-vit-base or similar weight repo
            hf_hub_download(
                repo_id="ybelkada/segment-anything",
                filename="checkpoints/sam_vit_b_01ec64.pth",
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            
            source_sam = models_dir / "checkpoints" / "sam_vit_b_01ec64.pth"
            if source_sam.exists():
                shutil.move(str(source_sam), str(target_sam))
                if (models_dir / "checkpoints").exists():
                    shutil.rmtree(models_dir / "checkpoints")
                print(f"   Moved to: {target_sam}")
        except Exception as e:
            print(f"⚠ SAM download failed: {e}")
    else:
        print(f"✓ Already exists: {target_sam}")

    # ------------------------------------------------------------------
    # 6. BetterThanWords LoRA (SDXL)
    # ------------------------------------------------------------------
    print("\n[6/6] Downloading BetterThanWords LoRA (SDXL)...")
    loras_dir = models_dir / "loras"
    loras_dir.mkdir(exist_ok=True)
    target_lora = loras_dir / "BetterThanWords-merged-SDXL-LoRA-v3.safetensors"

    if not target_lora.exists():
        try:
            # Download from lllyasviel/fav_models
            hf_hub_download(
                repo_id="lllyasviel/fav_models",
                filename="fav/BetterThanWords-merged-SDXL-LoRA-v3.safetensors",
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            
            source_lora = models_dir / "fav" / "BetterThanWords-merged-SDXL-LoRA-v3.safetensors"
            if source_lora.exists():
                shutil.move(str(source_lora), str(target_lora))
                if (models_dir / "fav").exists():
                    shutil.rmtree(models_dir / "fav")
                print(f"   Moved to: {target_lora}")
        except Exception as e:
            print(f"⚠ LoRA download failed: {e}")
    else:
        print(f"✓ Already exists: {target_lora}")

    print("\n" + "="*50)
    print("ALL DOWNLOADS COMPLETE")
    print("="*50)
    print("Models are located in:", models_dir)
    print("\nPaths found:")
    print(f"1. DRESS_SEG_MODEL_PATH: {target_b2}")
    print(f"2. MASKING_SEG_MODEL_PATH: {segformer_b3_path}")
    print(f"3. MASKING_Onnx_MODEL_PATH: {target_path}")
    print(f"4. MOONDREAM_MODEL_PATH: {moondream_path}")
    print(f"5. SAM_MODEL_PATH: {target_sam}")
    print(f"6. LORA_BTW_PATH: {target_lora}")
    print("="*50)

if __name__ == "__main__":
    download_models()
