import subprocess
import sys
import os
import importlib.metadata
import torch
import time
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path

def check_and_install_dependencies():
    """Ensure strictly compatible dependencies are installed at runtime."""
    # Moondream2 works best with modern transformers (no strict pinning for 2025 revision)
    required_packages = [
        ("transformers", None),
        ("accelerate", None),
        ("einops", None),
        ("timm", None)
    ]
    
    needs_restart = False
    install_list = []

    for package, version in required_packages:
        try:
            installed_version = importlib.metadata.version(package)
            if version and installed_version != version:
                print(f"🌙 Moondream: {package} version mismatch ({installed_version} != {version})")
                install_list.append(f"{package}=={version}")
                needs_restart = True
        except importlib.metadata.PackageNotFoundError:
            print(f"🌙 Moondream: {package} missing.")
            install_list.append(f"{package}=={version}" if version else package)
            needs_restart = True

    if needs_restart:
        print(f"🌙 Moondream: Auto-installing dependencies: {', '.join(install_list)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + install_list)
            
            # Simple and robust discovery for launch.py relative to this file
            # /Fooocus/modules/moondream_helper.py -> /Fooocus/launch.py
            current_file = Path(__file__).resolve()
            launcher_path = current_file.parent.parent / "launch.py"
            
            if not launcher_path.exists():
                # Fallback: search up
                temp_p = current_file
                for _ in range(3):
                    if (temp_p.parent / "launch.py").exists():
                        launcher_path = temp_p.parent / "launch.py"
                        break
                    temp_p = temp_p.parent

            print(f"🔄 Moondream: Restarting application to apply changes...")
            print(f"   Launcher found at: {launcher_path}")
            
            # Use absolute path for both executable and script to be safe
            os.execv(sys.executable, [sys.executable, str(launcher_path.absolute())] + sys.argv[1:])
        except Exception as e:
            print(f"❌ Moondream: Failed to install/restart: {e}")
            import traceback
            traceback.print_exc()

# Run check immediately on import
check_and_install_dependencies()

# Module-level globals for lazy loading
_model = None
_processor = None

def get_moondream_model_path():
    """Locate the Moondream2 model in the Fooocus structure."""
    # Try relative to this file
    base_dir = Path(__file__).resolve().parents[1]
    local_path = base_dir / "models" / "moondream2"
    
    if local_path.exists():
        return str(local_path)
    
    # Try one more fallback: shared models dir
    shared_path = Path("models/moondream2").absolute()
    if shared_path.exists():
        return str(shared_path)

    print(f"🌙 Moondream: Local model not found at {local_path}. Falling back to HuggingFace...")
    return "vikhyatk/moondream2"

def load_moondream():
    """Load the Moondream2 model and processor into memory."""
    global _model, _processor
    
    if _model is not None:
        return _model, _processor
    
    model_id = get_moondream_model_path()
    print(f"🌙 Moondream: Loading model from {model_id}...")
    
    try:
        # Use proven stable revision
        revision = "2024-04-02"
        
        _processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            revision=revision
        ).to(device)
        
        _model.eval()
        print(f"✅ Moondream: Loaded successfully on {device}")
        return _model, _processor
        
    except Exception as e:
        print(f"❌ Moondream: Failed to load: {e}")
        return None, None

def analyze_cloth(image):
    """
    Analyze a cloth image and return a detailed prompt description.
    Args:
        image: A PIL Image or numpy array.
    """
    if image is None:
        return ""
        
    # Lazy load the model
    model, processor = load_moondream()
    if model is None:
        return "Error: Moondream model could not be loaded."
        
    # Convert numpy to PIL if necessary
    if not isinstance(image, Image.Image):
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            return "Error: Unsupported image type."
            
    # Ensure RGB
    image = image.convert("RGB")
    
    query = (
        "Analyze the outfit in detail. "
        "Provide reasoning (colors, fabric, style, fit, accessories)."
    )
    
    print(f"🌙 Moondream: Querying (Reasoning Mode): {query}...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            image_embeds = model.encode_image(image)
            response = model.answer_question(
                image_embeds=image_embeds,
                question=query,
                tokenizer=processor
            )
            
        elapsed = time.time() - start_time
        description = response.strip()
        print(f"✅ Moondream: Analysis complete in {elapsed:.2f}s")
        return description
        
    except Exception as e:
        print(f"❌ Moondream: Error during analysis: {e}")
        return f"Error during cloth analysis: {str(e)}"
