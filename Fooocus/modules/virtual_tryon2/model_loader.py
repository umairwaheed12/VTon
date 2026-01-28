import os
import torch
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import AutoProcessor, AutoModelForCausalLM
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import time

# Global Cache
_B2_SESSION = None
_B3_MODEL = None
_B3_PROCESSOR = None
_SAM_PREDICTOR = None
_POSE_DETECTOR = None
_LIP_SESSION = None
_MOONDREAM_MODEL = None
_MOONDREAM_PROCESSOR = None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def find_b2_model(models_dir):
    """Robustly find the B2 ONNX model in common locations."""
    # 1. Direct in models/ (User's current structure)
    path1 = os.path.join(models_dir, 'segformer_b2_clothes.onnx')
    if os.path.exists(path1):
        return path1
    
    # 2. In SegFormerB2Clothes/ (models_downloader structure)
    path2 = os.path.join(models_dir, 'SegFormerB2Clothes', 'segformer_b2_clothes.onnx')
    if os.path.exists(path2):
        return path2
        
    return None

def get_b2_session(model_path):
    global _B2_SESSION
    if _B2_SESSION is None:
        # If absolute path fails, try searching relative to models dir
        if not os.path.exists(model_path):
            models_dir = os.path.dirname(model_path)
            found_path = find_b2_model(models_dir)
            if found_path:
                model_path = found_path
        
        print(f"🔄 Loader: Loading SegFormer B2 (ONNX) from {model_path}...")
        available = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        if 'CUDAExecutionProvider' not in providers[0]:
             print(f"⚠️ Loader: CUDAExecutionProvider not available in ONNX! Available: {available}. Using CPU for B2.")
        
        try:
            _B2_SESSION = ort.InferenceSession(str(model_path), providers=providers)
            print(f"✅ Loader: SegFormer B2 loaded. (Provider: {_B2_SESSION.get_providers()[0]})")
        except Exception as e:
            print(f"❌ Loader: Failed to load SegFormer B2: {e}")
            raise e
    return _B2_SESSION

def get_b3_model_and_processor(model_path):
    global _B3_MODEL, _B3_PROCESSOR
    if _B3_MODEL is None or _B3_PROCESSOR is None:
        print(f"🔄 Loader: Loading SegFormer B3 (Fashion) from {model_path}...")
        device = get_device()
        try:
            _B3_PROCESSOR = SegformerImageProcessor.from_pretrained(str(model_path))
            _B3_MODEL = SegformerForSemanticSegmentation.from_pretrained(str(model_path)).to(device)
            _B3_MODEL.eval()
            print(f"✅ Loader: SegFormer B3 loaded on {device}.")
        except Exception as e:
             print(f"❌ Loader: Failed to load SegFormer B3: {e}")
             raise e
    return _B3_MODEL, _B3_PROCESSOR

def get_sam_predictor(model_path):
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is None:
        print(f"🔄 Loader: Loading SAM (ViT-B) from {model_path}...")
        device = get_device()
        try:
            # Using vit_b as it matches the file found in models/sam
            sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
            sam.to(device=device)
            _SAM_PREDICTOR = SamPredictor(sam)
            print(f"✅ Loader: SAM loaded on {device}.")
        except Exception as e:
             print(f"❌ Loader: Failed to load SAM: {e}")
             raise e
    return _SAM_PREDICTOR

def get_lip_session(model_path):
    global _LIP_SESSION
    if _LIP_SESSION is None:
        print(f"🔄 Loader: Loading LIP Parsing (ONNX) from {model_path}...")
        available = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            _LIP_SESSION = ort.InferenceSession(str(model_path), providers=providers)
            print(f"✅ Loader: LIP Parsing loaded. (Provider: {_LIP_SESSION.get_providers()[0]})")
        except Exception as e:
             print(f"❌ Loader: Failed to load LIP Parsing: {e}")
             raise e
    return _LIP_SESSION

def get_pose_detector():
    global _POSE_DETECTOR
    if _POSE_DETECTOR is None:
        print("🔄 Loader: Initializing MediaPipe Pose...")
        # MediaPipe usually runs on CPU in Python, but it's lightweight.
        _POSE_DETECTOR = mp.solutions.pose.Pose(
            static_image_mode=True, 
            model_complexity=2, 
            min_detection_confidence=0.5
        )
        print("✅ Loader: MediaPipe Pose initialized.")
    return _POSE_DETECTOR

def get_moondream_model():
    from modules import moondream_helper
    return moondream_helper.load_moondream()

def preload_all_models(fooocus_root):
    """
    Call this at startup to load all models into memory/VRAM.
    """
    print("\n🚀 Loader: Starting Pre-load of All VTON Models on GPU...")
    start = time.time()
    
    # Paths
    models_dir = os.path.join(fooocus_root, 'models')
    
    # 1. SegFormer B2 (Clothes)
    b2_path = find_b2_model(models_dir)
    if b2_path:
        get_b2_session(b2_path)
    else:
        print(f"⚠️ Loader: B2 model (segformer_b2_clothes.onnx) not found in {models_dir} or subfolders!")

    # 2. SegFormer B3 (Fashion)
    b3_path = os.path.join(models_dir, 'segformer-b3-fashion')
    if os.path.exists(b3_path):
        get_b3_model_and_processor(b3_path)
    else:
        print(f"⚠️ Loader: B3 model not found at {b3_path}")
        
    # 3. SAM (ViT-B)
    sam_path = os.path.join(models_dir, 'sam', 'sam_vit_b_01ec64.pth')
    if os.path.exists(sam_path):
        get_sam_predictor(sam_path)
    else:
        print(f"⚠️ Loader: SAM model not found at {sam_path}")

    # 4. LIP Parsing
    lip_path = os.path.join(models_dir, 'humanparsing', 'parsing_lip.onnx')
    if os.path.exists(lip_path):
        get_lip_session(lip_path)
    else:
         print(f"⚠️ Loader: LIP model not found at {lip_path}")
         
    # 5. MediaPipe
    get_pose_detector()
    
    # 6. Moondream
    get_moondream_model()
    
    end = time.time()
    print(f"🏁 Loader: Pre-load complete in {end - start:.2f}s.\n")
