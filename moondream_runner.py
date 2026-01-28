# ===============================
# Auto-Install Dependencies
# ===============================
import subprocess
import sys
import os

def install_dependencies():
    required_packages = [
        "transformers", 
        "torch", 
        "torchvision", 
        "pillow", 
        "accelerate", 
        "einops"
    ]
    print("⏳ Checking and installing dependencies...")
    try:
        # Check if packages are installed by trying to import them involves more complexity,
        # so we'll just pip install them. Pip is good at skipping if already satisfied.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required_packages)
        print("✅ Dependencies installed.")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

# Run installation before other imports
try:
    import transformers
    import torch
    import accelerate
    import PIL
except ImportError:
    install_dependencies()

# ===============================
# Imports
# ===============================
import time
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ===============================
# Load Model & Processor
# ===============================
model_name = "vikhyatk/moondream2"

print(f"⏳ Loading {model_name}...")
try:
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"👉 Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to("cpu")
        
    model.eval()
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ===============================
# Load Image (Auto Download for Test)
# ===============================
# Using a standard test image if no path is provided or if the path is invalid
image_path = "/content/image (22).png" # Default from snippet

# Download a test image if the specified file doesn't exist
if not os.path.exists(image_path):
    print(f"⚠️ Image not found at {image_path}. Downloading a sample image for testing...")
    url = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg" # Example image from Moondream repo
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("✅ Sample image downloaded.")
    except Exception as e:
        print(f"❌ Failed to download sample image: {e}")
        sys.exit(1)
else:
    print(f"✅ Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")

# ===============================
# Encode Image
# ===============================
try:
    with torch.no_grad():
        image_embeds = model.encode_image(image)
except Exception as e:
    print(f"❌ Error encoding image: {e}")
    sys.exit(1)

# ===============================
# Query + Reasoning + Answer
# ===============================
query = (
    "Analyze the outfit in detail. "
    "Provide reasoning(colors, fabric, style, fit, accessories), "
)

print(f"❓ Query: {query}")
start_time = time.time()

try:
    response = model.answer_question(
        image_embeds=image_embeds,
        question=query,
        tokenizer=processor,
        # reasoning=True # Note: 'reasoning' arg might depend on specific version/revision, checking standard usage
    )
except TypeError:
    # Fallback if 'reasoning' argument is not supported in the loaded revision
    print("⚠️ 'reasoning' argument might not be supported directly, retrying without it...")
    response = model.answer_question(
        image_embeds=image_embeds,
        question=query,
        tokenizer=processor
    )

elapsed = time.time() - start_time

# ===============================
# Output
# ===============================
print("\n🧥 Outfit Analysis + Answer:")
print(response.strip())
print(f"\n⏱️ Generation time: {elapsed:.2f} seconds")
