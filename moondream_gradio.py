# ===============================
# Auto-Install Dependencies
# ===============================
import subprocess
import sys
import os

import importlib.metadata

def check_and_install_dependencies():
    # unexpected-keyword-arg-trust_remote_code issues usually fixed by using correct transformers version
    # moondream2 (revision 2024-04-02) works best with transformers==4.39.0
    required_packages = [
        ("transformers", "4.39.0"), 
        ("torch", None), 
        ("torchvision", None), 
        ("pillow", None), 
        ("accelerate", None), 
        ("einops", None),
        ("gradio", None)
    ]
    
    needs_restart = False
    install_list = []

    print("⏳ Verifying dependencies...")
    for package, version in required_packages:
        try:
            installed_version = importlib.metadata.version(package)
            if version and installed_version != version:
                print(f"⚠️  {package} version mismatch: found {installed_version}, need {version}")
                install_list.append(f"{package}=={version}")
                needs_restart = True
        except importlib.metadata.PackageNotFoundError:
            print(f"⚠️  {package} not found")
            install_list.append(f"{package}=={version}" if version else package)
            needs_restart = True

    if needs_restart:
        print(f"⏳ Installing/Upgrading: {', '.join(install_list)}...")
        try:
            # Install required packages
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + install_list)
            print("✅ Dependencies installed.")
            
            print("🔄 Restarting script to apply changes...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            sys.exit(1)
    else:
        print("✅ All dependencies match requirements.")

# Run dependency check before imports
check_and_install_dependencies()

import transformers
import torch
import accelerate
import PIL
import gradio as gr

# ===============================
# Imports
# ===============================
import time
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ===============================
# Load Model & Processor
# ===============================
model_name = "vikhyatk/moondream2"
# Check for local model from models_downloader.py
local_model_path = os.path.join(os.path.dirname(__file__), "models", "moondream2")
if os.path.exists(local_model_path):
    print(f"📂 Found local model at: {local_model_path}")
    model_name = local_model_path

model = None
processor = None

def load_model():
    global model, processor
    if model is not None:
        return
    
    print(f"⏳ Loading {model_name}...")
    try:
        # Using revision "2024-04-02" which is a known stable version of moondream2 
        # that avoids the 'PhiConfig' and 'all_tied_weights_keys' issues.
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision="2024-04-02"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"👉 Using device: {device}")

        # Note: device_map="auto" can cause 'all_tied_weights_keys' error with some custom models/accelerate versions
        # Using manual .to(device) is safer for this specific model.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            revision="2024-04-02"
        )
        model = model.to(device)
            
        model.eval()
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# ===============================
# Inference Function
# ===============================
def analyze_image(image, query):
    if image is None:
        return "Please upload an image."
    
    if model is None:
        load_model()
    
    # Ensure image is RGB
    image = image.convert("RGB")
    
    print(f"❓ Processing query: {query}")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            image_embeds = model.encode_image(image)
            
        try:
            # Attempt with reasoning if supported/desired, though standard usually just takes tokenizer
            # For simplicity and robustness with current versions, following standard flow
            response = model.answer_question(
                image_embeds=image_embeds,
                question=query,
                tokenizer=processor
            )
        except Exception as e:
            return f"Error during generation: {str(e)}"

        elapsed = time.time() - start_time
        result = f"{response.strip()}\n\n(Generation time: {elapsed:.2f} s)"
        return result
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

# ===============================
# Gradio Interface
# ===============================
def main():
    # Load model on startup
    load_model()
    
    default_query = (
        "Analyze the outfit in detail. "
        "Provide reasoning(colors, fabric, style, fit, accessories), "
    )

    with gr.Blocks(title="Moondream2 Outfit Analyzer") as demo:
        gr.Markdown("# 🌙 Moondream2 Outfit Analyzer")
        gr.Markdown("Upload an image to get a detailed outfit description.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                input_query = gr.Textbox(
                    label="Query", 
                    value=default_query,
                    lines=3
                )
                analyze_btn = gr.Button("Analyze Outfit", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Analysis Result", lines=10)
        
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, input_query],
            outputs=output_text
        )
    
    print("🚀 Launching Gradio app...")
    # server_name="0.0.0.0" makes it accessible on network (needed for RunPod/Container)
    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main()
