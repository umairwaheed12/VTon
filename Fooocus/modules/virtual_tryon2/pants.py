import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path
from modules.virtual_tryon2.vton_masking_helper import VTONMasker
from modules.virtual_tryon2.artificial_skin_helper import ArtificialSkinHelper

class LBSPantsWarper:
    def __init__(self, seg_model_path, b3_path=None):
        # Discover project root (4 levels up from Modules/Virtual_Tryon2)
        self.base_dir = Path(__file__).resolve().parents[3]
        self.seg_model_path = seg_model_path
        self.b3_path = b3_path if b3_path else self.base_dir / "models" / "segformer-b3-fashion"
        self._init_segformer()
        self._init_b3()
        self._init_mediapipe()
        self._init_lip()
        
        # Initialize Unified Masker
        sam_path = self.base_dir / "models" / "sam" / "sam_vit_b_01ec64.pth"
        self.masker = VTONMasker(
            seg_model_path=seg_model_path, 
            sam_model_path=str(sam_path) if sam_path.exists() else None
        )
        
        # Artificial Skin Helper
        self.skin_helper = ArtificialSkinHelper()
        
    def _init_lip(self):
        """Initialize LIP Human Parsing Model."""
        self.lip_model_path = self.base_dir / "models" / "humanparsing" / "parsing_lip.onnx"
        print(f"Loading LIP Model from {self.lip_model_path}...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.lip_session = ort.InferenceSession(self.lip_model_path, providers=providers)
            self.lip_input_name = self.lip_session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load LIP: {e}")
            self.lip_session = None
        
    def _init_b3(self):
        """Initialize B3 for Shorts detection."""
        print(f"Loading B3 Model from {self.b3_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.b3_processor = SegformerImageProcessor.from_pretrained(self.b3_path)
            self.b3_model = SegformerForSemanticSegmentation.from_pretrained(self.b3_path)
            self.b3_model.to(self.device).eval()
        except Exception as e:
            print(f"Failed to load B3: {e}")
            self.b3_model = None
        
    def _init_segformer(self):
        if not Path(self.seg_model_path).exists():
            raise FileNotFoundError(f"SegFormer model not found at {self.seg_model_path}")
        self.session = ort.InferenceSession(str(self.seg_model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def _init_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    # -----------------------------------------------------------
    # Pose & Seg
    # -----------------------------------------------------------
    def get_pose(self, image):
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if not results.pose_landmarks:
            return None
        
        lms = results.pose_landmarks.landmark
        mapping = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            # Hands for trimming
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22
        }
        kp = {}
        for name, idx in mapping.items():
            if idx < len(lms):
                kp[name] = np.array([lms[idx].x * w, lms[idx].y * h], dtype=np.float32)
        return kp

    def get_person_mask_lip(self, image):
        """
        Generates a High-Quality, smooth mask using LIP human parsing.
        Uses bilinear interpolation on logits to avoid "zig-zag" pixelation.
        """
        if self.lip_session is None:
            return None
            
        h, w = image.shape[:2]
        # Preprocess to LIP standard size
        img_resized = cv2.resize(image, (473, 473))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Float32 normalized [1, 3, 473, 473]
        img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # Inference
        print("🪄 Running LIP Human Parsing (Smooth Mode)...")
        outputs = self.lip_session.run(None, {self.lip_input_name: img_tensor})
        logits = outputs[0][0] # [20, 473, 473]
        
        # --- FIX ZIG ZAG: Resize logits smoothly BEFORE argmax ---
        # OpenCV resize expects (W, H)
        logits_smooth = np.zeros((20, h, w), dtype=np.float32)
        for i in range(20):
            logits_smooth[i] = cv2.resize(logits[i], (w, h), interpolation=cv2.INTER_LINEAR)
            
        # Argmax on full-res smooth logits
        pred_smooth = np.argmax(logits_smooth, axis=0).astype(np.uint8)
        
        return pred_smooth

    def get_garment_mask_hybrid(self, image):
        """
        Hybrid Detection:
        1. Check B3 for Shorts (Class 8).
        2. If Shorts -> Return B3 Mask (Classes 8, 33, 7) & Type 'shorts'.
        3. Else -> Fallback to B2 for Pants/Skirts.
           - IF Skirt: Merge B2 Skirt Mask + B3 Skirt Mask to avoid holes.
        """
        img_h, img_w = image.shape[:2]
        
        # 1. B3 Inference (Shorts Check & Skirt Prep)
        b3_seg = None
        is_shorts = False
        shorts_mask = None
        
        if self.b3_model:
            # --- FIX: Convert to RGB for B3 ---
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.b3_processor(images=image_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.b3_model(**inputs)
                logits = outputs.logits
            
            upsampled = torch.nn.functional.interpolate(logits, size=(img_h, img_w), mode='bilinear', align_corners=False)
            b3_seg = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            
            # Check for Shorts (8), Pants (7), and Skirt (9)
            count_shorts = np.sum(b3_seg == 8)
            count_pants_b3 = np.sum(b3_seg == 7)
            count_skirt_b3 = np.sum(b3_seg == 9)
            count_total_leg = count_shorts + count_pants_b3 + count_skirt_b3
            
            # --- USER REQUEST: INCLUDE DECORATIVE LABELS ---
            # 35: buckle, 37: applique, 42: ribbon, 44: ruffle, 45: sequin
            DECO_CLASSES = [35, 37, 42, 44, 45]
            ACCESSORY_CLASSES = [20, 33] # Belt, Pocket
            
            count_deco = np.sum(np.isin(b3_seg, DECO_CLASSES))
            
            # --- HYBRID DETECTION (Pants + Skirt = Decorative) ---
            # If both pants and skirt are significant, it's the "4th case" (Decorative/Hybrid)
            is_hybrid = count_pants_b3 > 500 and count_skirt_b3 > 500
            
            # New Condition: Pants (7) AND Shorts (8) -> Treat as SHORTS
            has_pants_and_shorts = count_pants_b3 > 200 and count_shorts > 200

            # Heuristic: B3 is generally better. If we find legwear or deco, use it.
            if count_total_leg > 500 or count_deco > 300:
                print(f"👖 B3 Detected: (Pants: {count_pants_b3}, Skirt: {count_skirt_b3}, Shorts: {count_shorts}, Deco: {count_deco})")
                
                if count_deco > 300 or is_hybrid:
                    if is_hybrid:
                        print("   ✨ Classified as DECORATIVE (Hybrid Pants/Skirt detected)")
                    else:
                        print("   ✨ Classified as DECORATIVE (Ornamental labels detected)")
                    g_type = 'decorative'
                    mask_indices = [7, 8, 9] + ACCESSORY_CLASSES + DECO_CLASSES
                    is_shorts = False

                elif has_pants_and_shorts:
                   # Pants + Shorts = Shorts (User Request)
                   is_shorts = True
                   g_type = 'shorts'
                   print("   -> Classified as SHORTS (Pants+Shorts detected)")
                   mask_indices = [7, 8] + ACCESSORY_CLASSES + [35]

                elif count_shorts > count_pants_b3 and count_shorts > count_skirt_b3:
                    is_shorts = True
                    g_type = 'shorts'
                    print("   -> Classified as SHORTS")
                    mask_indices = [7, 8] + ACCESSORY_CLASSES + [35]
                elif count_skirt_b3 > count_pants_b3:
                    is_shorts = False
                    g_type = 'skirt'
                    print("   -> Classified as SKIRT (from B3)")
                    mask_indices = [9] + ACCESSORY_CLASSES + [35]
                else:
                    is_shorts = False
                    g_type = 'pants'
                    print("   -> Classified as PANTS")
                    mask_indices = [7, 8] + ACCESSORY_CLASSES + [35]
                
                # Create Mask
                mask_b3 = np.isin(b3_seg, mask_indices).astype(np.uint8) * 255
                
                # --- COMPONENT ISOLATION (Fix for Oversized Outfits) ---
                # Goal: If the image has both shirt and pants, and accessories/pockets on shirt leaked in,
                # isolate ONLY the largest bottom-most component (the pants/skirt).
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_b3, connectivity=8)
                if num_labels > 1:
                    valid_stats = stats[1:]
                    areas = valid_stats[:, 4]
                    large_indices = np.where(areas > 800)[0] + 1
                    
                    if len(large_indices) > 0:
                        # Pick the BOTTOM-MOST component (highest 'top' + 'height' value)
                        # This effectively targets the legs.
                        bottoms = stats[large_indices, 1] + stats[large_indices, 3]
                        bottom_most_idx = large_indices[np.argmax(bottoms)]
                        
                        isolated_mask = np.zeros_like(mask_b3)
                        isolated_mask[labels == bottom_most_idx] = 255
                        mask_b3 = isolated_mask
                        print(f"   ✓ Isolated Lower Garment Component (Area: {stats[bottom_most_idx, 4]})")
                # -------------------------------------------------------------

                # Clean
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_b3 = cv2.morphologyEx(mask_b3, cv2.MORPH_OPEN, kernel)
                
                return mask_b3, g_type
            
        # 2. Fallback to B2 (If B3 failed or found nothing)
        print("⚠ B3 didn't see pants/shorts, falling back to B2...")
        mask_b2, g_type = self.get_garment_mask_b2(image)
        
        # 3. ENHANCE SKIRT with B3 (If B2 found skirt)
        if g_type == 'skirt' and b3_seg is not None:
            print("👗 Enhancing Skirt Mask with B3...")
            
            # Find 'Skirt' ID in B3 config
            skirt_id = -1
            for id_str, label in self.b3_model.config.id2label.items():
                if 'skirt' in label.lower():
                    skirt_id = int(id_str)
                    print(f"   Found B3 Skirt Label: '{label}' (ID: {skirt_id})")
                    break
            
            if skirt_id != -1:
                # Extract B3 Skirt Mask
                # Also include 'dress' (often 7?) if needed, but let's stick to 'skirt' first.
                # Actually, sometimes skirts are detected as connected parts.
                # Let's strictly use the ID found.
                mask_b3 = (b3_seg == skirt_id).astype(np.uint8) * 255
                
                # UNION: B2 | B3
                combined_mask = cv2.bitwise_or(mask_b2, mask_b3)
                
                # Calculate improvement
                # b2_pixels = cv2.countNonZero(mask_b2)
                # combined_pixels = cv2.countNonZero(combined_mask)
                # print(f"   Added {combined_pixels - b2_pixels} pixels from B3.")
                
                mask_b2 = combined_mask
                
        # Clean Final Mask (for Skirt/Pants)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_b2 = cv2.morphologyEx(mask_b2, cv2.MORPH_OPEN, kernel)
        
        return mask_b2, g_type

    def get_garment_mask_b2(self, image):
        """Original B2 logic for Pants(6) vs Skirt(5)."""
        img_h, img_w = image.shape[:2]
        inp_img = cv2.resize(image, (512, 512))
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp_img = (inp_img / 255.0 - mean) / std
        inp_img = inp_img.transpose(2, 0, 1)[None, ...].astype(np.float32)
        
        outputs = self.session.run(None, {self.input_name: inp_img})
        logits = outputs[0][0].transpose(1, 2, 0)
        logits = cv2.resize(logits, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        seg_map = np.argmax(logits, axis=2).astype(np.uint8)
        
        # Determine Garment Type
        pants_pixels = (seg_map == 6).astype(np.uint8) * 255
        skirt_pixels = (seg_map == 5).astype(np.uint8) * 255
        
        p_count = cv2.countNonZero(pants_pixels)
        s_count = cv2.countNonZero(skirt_pixels)
        
        g_type = 'skirt' if s_count > p_count else 'pants'
        mask = skirt_pixels if g_type == 'skirt' else pants_pixels
        
        # Clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Filter Noise
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            mask = np.zeros_like(mask)
            max_area = max(cv2.contourArea(c) for c in cnts)
            for c in cnts:
                if cv2.contourArea(c) > 0.05 * max_area:
                     cv2.drawContours(mask, [c], -1, 255, -1)
        
        return mask, g_type

    # -----------------------------------------------------------
    # Bone Logic
    # -----------------------------------------------------------
    def get_bones(self, pose):
        """Returns dict of bones: {name: (start_pt, end_pt)}"""
        bones = {
            'waist': (pose['left_hip'], pose['right_hip']),
            'l_thigh': (pose['left_hip'], pose['left_knee']),
            'l_shin': (pose['left_knee'], pose['left_ankle']),
            'r_thigh': (pose['right_hip'], pose['right_knee']),
            'r_shin': (pose['right_knee'], pose['right_ankle']),
        }
        return bones

    def compute_anisotropic_affine(self, src_start, src_end, dst_start, dst_end, width_scale):
        """
        Computes Matrix that maps src_segment to dst_segment.
        - Longitudinal scale = dst_len / src_len
        - Transverse scale = width_scale
        """
        src_vec = src_end - src_start
        dst_vec = dst_end - dst_start
        
        src_len = np.linalg.norm(src_vec)
        dst_len = np.linalg.norm(dst_vec)
        
        if src_len < 1e-6 or dst_len < 1e-6:
            return np.eye(2, 3, dtype=np.float32)
            
        scale_x = dst_len / src_len
        scale_y = width_scale
        
        # Angle of source vector
        angle_src = np.arctan2(src_vec[1], src_vec[0])
        # Angle of dest vector
        angle_dst = np.arctan2(dst_vec[1], dst_vec[0])
        
        # 1. Transform Src to Local (Aligned with X-axis)
        # Translate to origin
        T1 = np.array([[1, 0, -src_start[0]], 
                       [0, 1, -src_start[1]], 
                       [0, 0, 1]], dtype=np.float32)
        # Rotate by -angle_src
        c1, s1 = np.cos(-angle_src), np.sin(-angle_src)
        R1 = np.array([[c1, -s1, 0], 
                       [s1, c1, 0], 
                       [0, 0, 1]], dtype=np.float32)
                       
        # 2. Scale (Non-Uniform)
        S = np.array([[scale_x, 0, 0],
                      [0, scale_y, 0],
                      [0, 0, 1]], dtype=np.float32)
                      
        # 3. Transform Local to Dst
        # Rotate by angle_dst
        c2, s2 = np.cos(angle_dst), np.sin(angle_dst)
        R2 = np.array([[c2, -s2, 0], 
                       [s2, c2, 0], 
                       [0, 0, 1]], dtype=np.float32)
        # Translate to dst_start
        T2 = np.array([[1, 0, dst_start[0]], 
                       [0, 1, dst_start[1]], 
                       [0, 0, 1]], dtype=np.float32)
        
        # Combine: M = T2 @ R2 @ S @ R1 @ T1
        M = T2 @ R2 @ S @ R1 @ T1
        return M[:2, :]

    def get_inverse_affines(self, src_pose, dst_pose, width_mult=1.0):
        """
        Returns M_inv for each bone (Target -> Source).
        width_mult: Factor to widen the result (>1.0 = Wider, <1.0 = Slimmer)
        """
        src_bones = self.get_bones(src_pose)
        dst_bones = self.get_bones(dst_pose)
        
        # Calculate global width scale (Hip Width Ratio)
        s_hip_w = np.linalg.norm(src_pose['left_hip'] - src_pose['right_hip'])
        d_hip_w = np.linalg.norm(dst_pose['left_hip'] - dst_pose['right_hip'])
        
        # Base Transverse Scale (Thickness/Height)
        base_scale = s_hip_w / (d_hip_w + 1e-6)
        
        transforms = {}
        for name in src_bones:
            if name in dst_bones:
                s_start, s_end = src_bones[name]
                d_start, d_end = dst_bones[name]
                
                # SPLIT LOGIC:
                # 1. Waist (Horizontal Bone): "Width" is along the bone (Longitudinal).
                #    We stretch the TARGET bone to make the texture map "wider".
                #    Transverse (Height) should NOT be widened.
                if name == 'waist':
                    # Stretch Target Bone Length by width_mult
                    center = (d_start + d_end) / 2.0
                    half_vec = (d_end - d_start) / 2.0
                    d_start_mod = center - half_vec * width_mult
                    d_end_mod = center + half_vec * width_mult
                    
                    # Pass base_scale (affects Height) unchanged
                    M_inv = self.compute_anisotropic_affine(d_start_mod, d_end_mod, s_start, s_end, base_scale)
                    
                # 2. Legs (Vertical/Diagonal Bones): "Width" is across the bone (Transverse).
                #    We scale the Transverse Scale factor.
                #    Longitudinal (Length) should NOT be changed.
                else:
                    # To Widen, we reduce the Inverse Scale (sample from smaller source width)
                    transverse_scale = base_scale / width_mult
                    M_inv = self.compute_anisotropic_affine(d_start, d_end, s_start, s_end, transverse_scale)
                
                transforms[name] = M_inv
                
        return transforms

    # -----------------------------------------------------------
    # Skinning Weights
    # -----------------------------------------------------------
    def dist_to_segment(self, p, a, b):
        ab = b - a
        len_sq = np.sum(ab**2)
        if len_sq == 0:
            return np.linalg.norm(p - a)
        t = np.clip(np.dot(p - a, ab) / len_sq, 0, 1)
        projection = a + t * ab
        return np.linalg.norm(p - projection)

    def compute_weights_map(self, shape, pose, bones_list, sigma=45.0, hard=False, return_dists=False):
        h, w = shape[:2]
        
        scale_factor = 0.5 
        small_h, small_w = int(h*scale_factor), int(w*scale_factor)
        sy, sx = np.mgrid[0:small_h, 0:small_w]
        
        grid_pts_small = np.stack([sx, sy], axis=2).reshape(-1, 2).astype(np.float32)
        grid_pts_small /= scale_factor
        
        bones_coords = self.get_bones(pose)
        small_weights = np.zeros((small_h * small_w, len(bones_list)), dtype=np.float32)
        
        # Track min distance for masking
        if return_dists:
            min_dists = np.full((small_h * small_w), np.inf, dtype=np.float32)

        for i, b_name in enumerate(bones_list):
            start, end = bones_coords[b_name]
            ab = end - start
            len_sq = np.sum(ab**2)
            if len_sq < 1e-6:
                dists = np.linalg.norm(grid_pts_small - start, axis=1)
            else:
                pa = grid_pts_small - start
                t = np.sum(pa * ab, axis=1) / len_sq
                t = np.clip(t, 0, 1)
                proj = start + t[:, None] * ab
                dists = np.linalg.norm(grid_pts_small - proj, axis=1)
            
            # Logits = -d^2 / 2sigma^2
            logits = -(dists**2) / (2 * sigma**2)
            small_weights[:, i] = logits
            
            if return_dists:
                min_dists = np.minimum(min_dists, dists)
            
        if hard:
            # Hard Skinning: Nearest Bone takes all
            max_indices = np.argmax(small_weights, axis=1)
            one_hot = np.zeros_like(small_weights)
            one_hot[np.arange(len(small_weights)), max_indices] = 1.0
            small_weights = one_hot
        else:
            # Soft Skinning (LBS)
            # Stable Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
            max_logits = np.max(small_weights, axis=1, keepdims=True)
            shifted_logits = small_weights - max_logits
            exp_weights = np.exp(shifted_logits)
            total_w = np.sum(exp_weights, axis=1, keepdims=True)
            small_weights = exp_weights / total_w
        
        small_weights = small_weights.reshape(small_h, small_w, len(bones_list))
        
        full_weights = []
        for i in range(len(bones_list)):
            # Use NEAREST for hard weights to avoid blurring edges back into soft
            interp = cv2.INTER_NEAREST if hard else cv2.INTER_LINEAR
            fw = cv2.resize(small_weights[:,:,i], (w, h), interpolation=interp)
            full_weights.append(fw)
            
        output = np.stack(full_weights, axis=2)
        
        if return_dists:
            min_dists = min_dists.reshape(small_h, small_w)
            full_dists = cv2.resize(min_dists, (w, h), interpolation=cv2.INTER_LINEAR)
            return output, full_dists
            
        return output

    # -----------------------------------------------------------
    # Main Warp (LBS)
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # Cleanup Helpers
    # -----------------------------------------------------------
    def clean_with_b3(self, warped_img):
        """
        Uses SegFormer B3 to filter the warped image, strictly keeping only 
        clothing parts (Pants, Shorts, Skirt, Belt, Pocket) and rejecting 
        artifacts (like Shoes, Background, etc).
        """
        if self.b3_model is None:
            print("⚠ B3 Model not loaded, skipping cleanup.")
            return warped_img
            
        h, w = warped_img.shape[:2]
        
        # 1. Prepare Input (Remove Alpha, Convert to RGB)
        # B3 expects regular RGB images. Background is black/transparent in warped_img.
        rgb_img = warped_img[:, :, :3]
        
        # 2. Inference
        try:
            inputs = self.b3_processor(images=rgb_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.b3_model(**inputs)
            
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            
            # 3. Create Keep Mask
            # 7: Pants, 8: Shorts, 9: Skirt, 20: Belt, 33: Pocket, 35: Buckle
            KEEP_CLASSES = {7, 8, 9, 20, 33, 35}
            keep_mask = np.zeros((h, w), dtype=np.uint8)
            
            for cls in KEEP_CLASSES:
                keep_mask[pred_seg == cls] = 255
                
            # Dilate slightly to include edges/anti-aliasing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            keep_mask = cv2.dilate(keep_mask, kernel, iterations=1)
            
            # 4. Apply to Alpha Channel
            # Intersect existing alpha with our keep mask
            current_alpha = warped_img[:, :, 3]
            new_alpha = cv2.bitwise_and(current_alpha, current_alpha, mask=keep_mask)
            
            # Update Alpha
            warped_img[:, :, 3] = new_alpha
            
            # Debug: what did we remove?
            removed_mask = cv2.bitwise_xor(current_alpha, new_alpha)
            if cv2.countNonZero(removed_mask) > 0:
                print(f"🧹 B3 Cleanup: Removed {cv2.countNonZero(removed_mask)} pixels (Artifacts/Shoes/Bg)")
                
        except Exception as e:
            print(f"⚠ B3 Cleanup Failed: {e}")
            
        return warped_img

    # -----------------------------------------------------------
    # Main Warp (LBS)
    # -----------------------------------------------------------
    def warp_pants_lbs(self, pants_img, src_pose, dst_pose, target_shape):
        h, w = target_shape[:2]
        bones_list = ['waist', 'l_thigh', 'l_shin', 'r_thigh', 'r_shin']
        
        transforms = self.get_inverse_affines(src_pose, dst_pose)
        print("Computing skinning weights...")
        weights_map = self.compute_weights_map((h, w), dst_pose, bones_list)
        
        y, x = np.mgrid[0:h, 0:w]
        ones = np.ones_like(x)
        P_dst = np.stack([x, y, ones], axis=0).reshape(3, -1) 
        
        P_src_accum = np.zeros((2, h*w), dtype=np.float32)
        
        for i, b_name in enumerate(bones_list):
            if b_name not in transforms:
                continue
            M_inv = transforms[b_name]
            P_transformed = M_inv @ P_dst
            w_flat = weights_map[:, :, i].reshape(-1)
            P_src_accum += P_transformed * w_flat
            
        map_x = P_src_accum[0, :].reshape(h, w).astype(np.float32)
        map_y = P_src_accum[1, :].reshape(h, w).astype(np.float32)
        
        print("Remapping image...")
        warped = cv2.remap(pants_img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        # Apply B3 Cleanup to remove artifacts (shoes, ghost legs, etc)
        warped = self.clean_with_b3(warped)
        
        return warped

    def warp_pants_crossed(self, pants_img, src_pose, dst_pose, target_shape):
        """Dedicated Warper for Crossed Legs with Sharper Weights."""
        h, w = target_shape[:2]
        bones_list = ['waist', 'l_thigh', 'l_shin', 'r_thigh', 'r_shin']
        
        # INCREASE WIDTH for Crossed Legs
        # User requested "little bit much" -> 1.15x Wider
        width_mult = 1.15
        transforms = self.get_inverse_affines(src_pose, dst_pose, width_mult=width_mult)
        
        # SHARP SIGMA for Crossed Legs
        sigma = 8.0 
        print(f"❌ Warping Crossed Legs with Sigma={sigma} (Hard Skinning), Width={width_mult}x...")
        
        # Use HARD SKINNING -> No Ghosting!
        # Also return Dist Map for "Distance Masking" (Trims Far Artifacts)
        weights_map, dist_map = self.compute_weights_map((h, w), dst_pose, bones_list, sigma=sigma, hard=True, return_dists=True)
        
        y, x = np.mgrid[0:h, 0:w]
        ones = np.ones_like(x)
        P_dst = np.stack([x, y, ones], axis=0).reshape(3, -1) 
        
        P_src_accum = np.zeros((2, h*w), dtype=np.float32)
        
        for i, b_name in enumerate(bones_list):
            if b_name not in transforms:
                continue
            M_inv = transforms[b_name]
            P_transformed = M_inv @ P_dst
            w_flat = weights_map[:, :, i].reshape(-1)
            P_src_accum += P_transformed * w_flat
            
        map_x = P_src_accum[0, :].reshape(h, w).astype(np.float32)
        map_y = P_src_accum[1, :].reshape(h, w).astype(np.float32)
        
        warped = cv2.remap(pants_img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        
        # -----------------------------------------------------
        # ADAPTIVE DISTANCE MASKING (Smart Trim)
        # -----------------------------------------------------
        l_hip = dst_pose['left_hip']
        r_hip = dst_pose['right_hip']
        hip_width = np.linalg.norm(l_hip - r_hip)
        
        # A. Per-Bone Thresholds
        # We need "Loose" fit for Hips/Thighs (to save crotch)
        # We need "Tight" fit for Shins (to kill bottom flaps)
        
        # Identify "Owner" bone for each pixel (Hard Skinning was used, so argmax matches)
        owner_indices = np.argmax(weights_map, axis=2)
        
        # Bones: 0:Waist, 1:L_Thigh, 2:L_Shin, 3:R_Thigh, 4:R_Shin
        # Default: Loose (1.1x)
        thresh_map = np.full((h, w), hip_width * 1.1, dtype=np.float32)
        
        # Tighten for Shins (Indices 2 and 4)
        shin_mask = (owner_indices == 2) | (owner_indices == 4)
        thresh_map[shin_mask] = hip_width * 0.6 # Aggressive Trim at bottom
        
        # Apply Adaptive Distance Mask
        valid_mask = (dist_map < thresh_map).astype(np.uint8) * 255
        
        # B. WAIST HEIGHT CONSTRAINT (Relaxed)
        # Fix Regression: Use 0.5 offset to protect Crotch/Zipper
        waist_idx = 0 
        waist_influence = weights_map[:, :, waist_idx] > 0.5
        max_hip_y = max(l_hip[1], r_hip[1])
        crotch_limit_y = max_hip_y + hip_width * 0.5 # Relaxed Limit
        
        y_grid, _ = np.mgrid[0:h, 0:w]
        bad_waist_pixels = (waist_influence) & (y_grid > crotch_limit_y)
        valid_mask[bad_waist_pixels] = 0
        
        # C. ISLAND REMOVAL (Connected Components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(valid_mask, connectivity=8)
        
        if num_labels > 1:
            areas = stats[1:, 4]
            if len(areas) > 0:
                max_area = np.max(areas)
                for i in range(1, num_labels):
                    area = stats[i, 4]
                    if area < max_area * 0.2: 
                        valid_mask[labels == i] = 0
                        
        # Apply cleanup morph
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply to Alpha channel
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], warped[:, :, 3], mask=valid_mask)
        
        print(f"✂️  Applied Adaptive Mask (Hips=1.1, Shins=0.6) & Island Removal.")
        
        return warped

    def is_legs_crossed(self, pose):
        """Detects if legs are crossed based on Ankle X-coordinates swapping relative to Hips."""
        # Standard: Right (Img Left, Small X) < Left (Img Right, Big X)
        if 'left_ankle' not in pose or 'right_ankle' not in pose:
             return False
        
        # Check X order
        r_x = pose['right_ankle'][0]
        l_x = pose['left_ankle'][0]
        
        print(f"DEBUG CHECK: R_Ankle_X={r_x:.1f}, L_Ankle_X={l_x:.1f}")
        
        # If Right Ankle is to the RIGHT of Left Ankle (r_x > l_x) -> Crossed!
        if r_x > l_x:
            return True
            
        # Also check Knees for tight cross
        if 'left_knee' in pose and 'right_knee' in pose:
            rk_x = pose['right_knee'][0]
            lk_x = pose['left_knee'][0]
            print(f"DEBUG CHECK: R_Knee_X={rk_x:.1f}, L_Knee_X={lk_x:.1f}")
            if rk_x > lk_x:
                return True
                
        return False

    def warp_skirt_rigid(self, skirt_img, src_pose, dst_pose, target_shape, skirt_mask, garment_type='skirt'):
        """Natural Flow Affine Warp: Anchors waistband to 'High Hips' and flows hem to ankles."""
        h, w = target_shape[:2]
        
        # --- CONDITIONAL SIZING RATIOS ---
        # User wants Decorative to have skirt-top (high waist) but reach feet.
        lift_ratio = 0.35   # High waist for both skirt and decorative
        width_ratio = 1.35  # Flared for both skirt and decorative
        
        # 1. ANALYZE SOURCE SKIRT (Find Top-Corners and Bottom-Center)
        y_idx, x_idx = np.where(skirt_mask > 0)
        s_min_y, s_max_y = np.min(y_idx), np.max(y_idx)
        
        # Define Top Zone (Waistband) as top 5% of skirt height
        top_limit = s_min_y + int((s_max_y - s_min_y) * 0.05)
        top_mask = np.zeros_like(skirt_mask)
        top_mask[s_min_y:top_limit, :] = skirt_mask[s_min_y:top_limit, :]
        
        t_y, t_x = np.where(top_mask > 0)
        if len(t_x) == 0: # Safety fallback
             s_min_x, s_max_x = np.min(x_idx), np.max(x_idx)
             src_tl = [s_min_x, s_min_y]
             src_tr = [s_max_x, s_min_y]
        else:
             # Top-Left and Top-Right of the waistband blob
             s_tl_x = np.percentile(t_x, 5)
             s_tr_x = np.percentile(t_x, 95)
             src_tl = [s_tl_x, s_min_y]
             src_tr = [s_tr_x, s_min_y]

        # Source Bottom Center
        src_bc = [np.mean(x_idx), s_max_y]
        src_tri = np.float32([src_tl, src_tr, src_bc])
        
        # 2. DEFINE TARGET LANDMARKS (Person)
        l_hip = dst_pose['left_hip']
        r_hip = dst_pose['right_hip']
        hips_mid = (l_hip + r_hip) / 2.0
        
        # Waist Placement
        if 'left_shoulder' in dst_pose and 'right_shoulder' in dst_pose:
            l_sh = dst_pose['left_shoulder']
            r_sh = dst_pose['right_shoulder']
            sh_mid = (l_sh + r_sh) / 2.0
            
            torso_vec = hips_mid - sh_mid
            torso_len = np.linalg.norm(torso_vec)
            
            # Use dynamic lift_ratio
            lift_vec = -torso_vec / (torso_len + 1e-6)
            waist_center = hips_mid + lift_vec * (torso_len * lift_ratio)
        else:
            # Fallback
            hip_width = np.linalg.norm(r_hip - l_hip)
            waist_center = hips_mid.copy()
            waist_center[1] -= hip_width * (lift_ratio * 1.2) # Adjusted fallback lift
            
        # Target Waist Width
        hip_vec = r_hip - l_hip
        hip_w = np.linalg.norm(hip_vec)
        
        # Use dynamic width_ratio
        target_w = hip_w * width_ratio
        
        # Build Top Points
        hip_dir = hip_vec / (hip_w + 1e-6)
        dst_tl = waist_center - hip_dir * (target_w * 0.5)
        dst_tr = waist_center + hip_dir * (target_w * 0.5)
        
        # B. Bottom Flow: "Natural not straight"
        # Target is the center between ankles (or knees if ankles missing) to follow leg line
        l_ankle = dst_pose.get('left_ankle', dst_pose['left_knee'])
        r_ankle = dst_pose.get('right_ankle', dst_pose['right_knee'])
        
        dst_bc = (l_ankle + r_ankle) / 2.0
        
        # Adjust length?
        # Skirt length is preserved relative to width ratio usually, but Affine takes care of mapping
        # If we map skirt_bottom to ankles, it becomes a maxi skirt.
        # User didn't specify length, but "natural flow".
        # Let's maintain relative scale or just map direction vector?
        # Mapping to ankles makes it long. Let's Map to Knees + 20%?
        # Better: Map orientation only, preserve Aspect Ratio scale?
        # NO, Affine dictates scale. If we map to ankles, we force length.
        
        # NEW STRATEGY for Bottom Point:
        # 1. Calc scale based on waist width
        src_w_width = src_tr[0] - src_tl[0]
        dst_w_width = np.linalg.norm(dst_tr - dst_tl)
        scale = dst_w_width / (src_w_width + 1e-6)
        
        # 2. Calc Src Height
        src_h = s_max_y - s_min_y
        target_len = src_h * scale
        
        # 3. Project downwards from waist center along the "Leg/Gravity Axis"
        waist_mid = (dst_tl + dst_tr) / 2.0
        ankles_mid = (l_ankle + r_ankle) / 2.0
        
        # Axis vector (normalized)
        leg_vec = ankles_mid - waist_mid
        leg_len = np.linalg.norm(leg_vec)
        if leg_len < 1e-6: leg_vec = [0, 1] 
        else: leg_vec /= leg_len
        
        # --- USER REQUEST: FORCE FULL LENGTH for Decorative ---
        if garment_type == 'decorative':
             # Stretch to feet (ankles)
             print("📏 Forcing Full Length for Decorative Garment (to feet/ankles)")
             dst_bc_final = waist_mid + leg_vec * (leg_len * 1.05) # Add 5% for coverage
        else:
             # Standard Skirt: Keep Aspect Ratio
             dst_bc_final = waist_mid + leg_vec * target_len
        
        dst_tri = np.float32([dst_tl, dst_tr, dst_bc_final])
        
        # 3. WARP
        M = cv2.getAffineTransform(src_tri, dst_tri)
        warped = cv2.warpAffine(skirt_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return warped


    def get_forward_affines(self, src_pose, dst_pose):
        """Returns M for each bone (Source -> Target)."""
        src_bones = self.get_bones(src_pose)
        dst_bones = self.get_bones(dst_pose)
        
        s_hip_w = np.linalg.norm(src_pose['left_hip'] - src_pose['right_hip'])
        d_hip_w = np.linalg.norm(dst_pose['left_hip'] - dst_pose['right_hip'])
        width_scale = d_hip_w / (s_hip_w + 1e-6)
        
        transforms = {}
        for name in src_bones:
            if name in dst_bones:
                s_start, s_end = src_bones[name]
                d_start, d_end = dst_bones[name]
                M = self.compute_anisotropic_affine(s_start, s_end, d_start, d_end, width_scale)
                transforms[name] = M
        return transforms

    def draw_skeleton(self, img, pose, color=(0, 255, 0), thickness=2):
        bones = [
            ('waist', 'left_hip', 'right_hip'),
            ('l_thigh', 'left_hip', 'left_knee'),
            ('l_shin', 'left_knee', 'left_ankle'),
            ('r_thigh', 'right_hip', 'right_knee'),
            ('r_shin', 'right_knee', 'right_ankle')
        ]
        vis = img.copy()
        for _, start, end in bones:
            if start in pose and end in pose:
                p1 = tuple(POSE_INT(pose[start]))
                p2 = tuple(POSE_INT(pose[end]))
                cv2.line(vis, p1, p2, color, thickness)
        for pt in pose.values():
            cv2.circle(vis, tuple(POSE_INT(pt)), 4, color, -1)
        return vis

    # -----------------------------------------------------------
    # Pose Estimation Fallback
    # -----------------------------------------------------------
    def estimate_pose_from_mask(self, mask, garment_type='pants'):
        """Estimates hip/knee/ankle points. For Shorts, creates VIRTUAL Full-Leg pose."""
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) == 0:
            return None
        
        min_y, max_y = np.min(y_idx), np.max(y_idx)
        min_x, max_x = np.min(x_idx), np.max(x_idx)
        h = max_y - min_y
        w = max_x - min_x
        
        mid_x = (min_x + max_x) // 2
        
        # Helper to find centroid in a ROI
        def get_centroid(y_start, y_end, x_start, x_end):
            roi = mask[y_start:y_end, x_start:x_end]
            sub_y, sub_x = np.where(roi > 0)
            if len(sub_y) == 0:
                # Fallback to center of ROI
                return np.array([x_start + (x_end - x_start)/2, y_start + (y_end - y_start)/2], dtype=np.float32)
            return np.array([x_start + np.mean(sub_x), y_start + np.mean(sub_y)], dtype=np.float32)
            
        kp = {}
        
        if garment_type == 'shorts':
            # VIRTUAL SKELETON STRATEGY
            # Shorts only cover top of thigh. We need to extrapolate where knees/ankles WOULD be
            # if this were full pants, so LBS maps the "short" texture to the "top" of target thigh.
            
            # 1. Hips (Top 30% of image)
            hip_y_limit = min_y + int(h * 0.4)
            kp['right_hip'] = get_centroid(min_y, hip_y_limit, min_x, mid_x)
            kp['left_hip'] = get_centroid(min_y, hip_y_limit, mid_x, max_x)
            
            # Widen Hips heuristic (same as pants)
            center_x = (kp['right_hip'][0] + kp['left_hip'][0]) / 2
            widen_factor = 1.1 
            kp['right_hip'][0] = center_x + (kp['right_hip'][0] - center_x) * widen_factor
            kp['left_hip'][0] = center_x + (kp['left_hip'][0] - center_x) * widen_factor
            
            # 2. Virtual Knees & Ankles (Extrapolated downwards)
            # Standard ratio: Hip-to-Knee ~= 1.5 * Hip_Width
            virtual_thigh_len = w * 1.5
            virtual_shin_len = w * 1.5
            
            # Hip Y levels
            r_hip_y = kp['right_hip'][1]
            l_hip_y = kp['left_hip'][1]
            
            # Virtual Knees
            kp['right_knee'] = np.array([kp['right_hip'][0], r_hip_y + virtual_thigh_len])
            kp['left_knee'] = np.array([kp['left_hip'][0], l_hip_y + virtual_thigh_len])
            
            # Virtual Ankles
            kp['right_ankle'] = np.array([kp['right_hip'][0], r_hip_y + virtual_thigh_len + virtual_shin_len])
            kp['left_ankle'] = np.array([kp['left_hip'][0], l_hip_y + virtual_thigh_len + virtual_shin_len])
            
            print(f"🩳 Generated Virtual Skeleton for LBS. Virtual Knee Y: {r_hip_y + virtual_thigh_len:.1f}")
            
        else:
            # NORMAL PANTS/SKIRT LOGIC (Within bbox)
            # Vertical Zones
            hip_y_end = min_y + int(h * 0.25)
            knee_y_start = min_y + int(h * 0.4)
            knee_y_end = min_y + int(h * 0.6)
            ankle_y_start = max_y - int(h * 0.15)
            
            # Right Leg (Image Left)
            kp['right_hip'] = get_centroid(min_y, hip_y_end, min_x, mid_x)
            kp['right_knee'] = get_centroid(knee_y_start, knee_y_end, min_x, mid_x)
            kp['right_ankle'] = get_centroid(ankle_y_start, max_y, min_x, mid_x)
            
            # Left Leg (Image Right)
            kp['left_hip'] = get_centroid(min_y, hip_y_end, mid_x, max_x)
            kp['left_knee'] = get_centroid(knee_y_start, knee_y_end, mid_x, max_x)
            kp['left_ankle'] = get_centroid(ankle_y_start, max_y, mid_x, max_x)
            
            # ADJUSTMENT: Widen the hips slightly
            center_x = (kp['right_hip'][0] + kp['left_hip'][0]) / 2
            widen_factor = 1.1 # Push out by 10%
            
            kp['right_hip'][0] = center_x + (kp['right_hip'][0] - center_x) * widen_factor
            kp['left_hip'][0] = center_x + (kp['left_hip'][0] - center_x) * widen_factor
        
        # ---------------------------------------------------------
        # TILT CORRECTION (Force Horizontal Waistband for Source)
        # ---------------------------------------------------------
        # Assume source pants/shorts are upright. Force Hip Ys to match.
        if 'left_hip' in kp and 'right_hip' in kp:
            avg_y = (kp['left_hip'][1] + kp['right_hip'][1]) / 2.0
            kp['left_hip'][1] = avg_y
            kp['right_hip'][1] = avg_y
            # Also adjust knees/ankles relative to new hip Y? 
            # Ideally yes, but correcting just the anchor (hips) solves 90% of waistband tilt.
            # Let's trust the lower body detection or maybe leave it.
            # If we move hips, we technically stretch/squash the thigh if we don't move knees.
            # But the shift is likely small (just correcting tilt).
            print(f"⚖️  Flattened Source Hips to Y={avg_y:.1f} to prevent tilt.")

        return kp

    def process(self, person_path, cloth_path, output_path, original_img=None, clean_img=None):
        if original_img is not None:
             print("ℹ LBSPantsWarper: Using passed Original Image.")
             person = original_img
        else:
             print(f"Loading {person_path}...")
             person = cv2.imread(person_path)

        if clean_img is not None:
            print("ℹ LBSPantsWarper: Using passed Clean Image.")
            person_clean = clean_img
        else:
            # --- CLOTH REMOVAL (Integrated) ---
            try:
                from cloth_remover import ClothRemover
                print("Initializing Cloth Remover (pants mode)...")
                remover = ClothRemover(self.seg_model_path)
                print("Removing original pants...")
                person_clean, _ = remover.remove_pants(person)
            except Exception as e:
                print(f"Warning: Cloth Removal Failed: {e}. Using original person image.")
                person_clean = person.copy()

        cloth = cv2.imread(cloth_path)
        
        print("Detecting pose...")
        p_pose = self.get_pose(person)
        if not p_pose:
             print("Person Pose detection failed.")
             return

        # Attempt cloth pose
        c_pose = self.get_pose(cloth)
        
        print(f"Segmenting {cloth_path}...")
        mask, garment_type = self.get_garment_mask_hybrid(cloth)
        
        # FORCE POSE ESTIMATION FOR SHORTS & PANTS (Ignore MediaPipe which might pick up Shirt)
        if garment_type in ['shorts', 'pants']:
            print(f"👖 {garment_type.capitalize()} detected: Forcing mask-based pose estimation to avoid Shirt interference.")
            c_pose = None

        # Fallback if cloth pose failed (e.g. flat lay) or was forced to None
        if not c_pose:
            print(f"Cloth pose not found (likely flat lay or forced). Estimating from {garment_type} mask...")
            c_pose = self.estimate_pose_from_mask(mask, garment_type)
            if not c_pose:
                print("Failed to estimate pose from mask (mask empty?).")
                return

        # ---------------------------------------------------------
        # SHORTS HIGH-WAIST ADJUSTMENT (Target Pose Modification)
        # ---------------------------------------------------------
        if garment_type == 'shorts':
            # We want shorts to sit at "True Waist" (High Waist), not low hips.
            # Same logic as Affine Skirt: 35-50% up the torso.
            if 'left_shoulder' in p_pose and 'right_shoulder' in p_pose:
                 l_sh = p_pose['left_shoulder']
                 r_sh = p_pose['right_shoulder']
                 l_hip = p_pose['left_hip']
                 r_hip = p_pose['right_hip']
                 
                 hips_mid = (l_hip + r_hip) / 2.0
                 sh_mid = (l_sh + r_sh) / 2.0
                 
                 torso_vec = hips_mid - sh_mid
                 torso_len = np.linalg.norm(torso_vec)
                 
                 # Lift vector (Up towards shoulders)
                 lift_vec = -torso_vec / (torso_len + 1e-6)
                 
                 # Move hips UP by 25% of torso (Little above Hips)
                 lift_amount = torso_len * 0.12
                 
                 print(f"⬆️ Lifting Shorts Pose by {lift_amount:.1f}px (High Waist)")
                 p_pose['left_hip'] += lift_vec * lift_amount
                 p_pose['right_hip'] += lift_vec * lift_amount
                 
                 # Shift entire leg structure up to prevent stretching
                 p_pose['left_knee'] += lift_vec * lift_amount
                 p_pose['right_knee'] += lift_vec * lift_amount
                 p_pose['left_ankle'] += lift_vec * lift_amount
                 p_pose['right_ankle'] += lift_vec * lift_amount

        rgba = cv2.cvtColor(cloth, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        
        # --- DYNAMIC WARPING LOGIC ---
        is_crossed = self.is_legs_crossed(p_pose)
        
        # Determine "Closeness" (Very close legs act like a skirt case)
        is_close = False
        if 'right_hip' in p_pose and 'left_hip' in p_pose and 'right_ankle' in p_pose and 'left_ankle' in p_pose:
            hip_width = abs(p_pose['right_hip'][0] - p_pose['left_hip'][0])
            ankle_dist = abs(p_pose['right_ankle'][0] - p_pose['left_ankle'][0])
            # If ankles are closer than 40% of hip width, consider them "very close"
            if ankle_dist < (hip_width * 0.4):
                is_close = True
                print(f"🦶 Legs are VERY CLOSE (Dist: {ankle_dist:.1f}, HipW: {hip_width:.1f}).")

        masker_g_type = garment_type # Default
        
        if garment_type == 'shorts':
            print("Warping Shorts (LBS with Virtual Skeleton)...")
            warped_rgba = self.warp_pants_lbs(rgba, c_pose, p_pose, person.shape)
        elif garment_type == 'skirt':
            print("Warping Skirt (Affine Flow)...")
            warped_rgba = self.warp_skirt_rigid(rgba, c_pose, p_pose, person.shape, mask, garment_type='skirt')
        elif garment_type == 'decorative':
            if is_crossed or is_close:
                # ACT LIKE SKIRT WARP but keep DECORATIVE SIZE
                print(f"Warping Decorative Garment (Affine Flow - Skirt Style for {'Crossed' if is_crossed else 'Close'} Legs)...")
                warped_rgba = self.warp_skirt_rigid(rgba, c_pose, p_pose, person.shape, mask, garment_type='decorative')
                masker_g_type = 'skirt' # Trigger aggressive 55/95 masking
            else:
                # ACT LIKE PANTS
                print("Warping Decorative Garment (LBS - Pants Style for Parted Legs)...")
                warped_rgba = self.warp_pants_lbs(rgba, c_pose, p_pose, person.shape)
                masker_g_type = 'decorative' # Trigger moderate 40/70 masking
        elif garment_type == 'pants':
            if is_crossed:
                print("🔀 Using Crossed Legs Warper...")
                warped_rgba = self.warp_pants_crossed(rgba, c_pose, p_pose, person.shape)
            else:
                print("Warping Pants (LBS)...")
                warped_rgba = self.warp_pants_lbs(rgba, c_pose, p_pose, person.shape)
        else:
            # Standard Skirt
            print("Warping Skirt (Affine Flow)...")
            warped_rgba = self.warp_skirt_rigid(rgba, c_pose, p_pose, person.shape, mask)
        
        # ----------------------------------------------------------------
        # HAND TRIMMING (User Request: "mask hands only and then trim the area")
        # 1. Create Hand Mask from Person Pose
        h, w = person.shape[:2]
        hand_trim_mask = np.zeros((h, w), dtype=np.uint8)
        hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]
        # We need the full landmarks for this
        if 'left_wrist' in p_pose or 'right_wrist' in p_pose:
            # Use vton_masking_helper to get the PRECISE hand-shaped mask
             hand_trim_mask = self.masker.get_precise_hand_mask(person)
             
             # Apply Trimming (Cut the warped garment where hands are)
             if np.any(hand_trim_mask > 0):
                   print("✂️  Trimming Garment with Precise Hand Shape...")
                   warped_rgba[:, :, 3][hand_trim_mask > 0] = 0
        # ----------------------------------------------------------------
        
        # --- UPPER GARMENT PROTECTION (User Request) ---
        # "perosn upper cloth does not hide behid teh pants"
        print("✂️  Protecting Upper Garment Visibility...")
        upper_cloth_mask = self.masker.get_upper_body_mask(person)
        if np.any(upper_cloth_mask > 0):
             # Trim the warped garment where upper body clothing is detected
             warped_rgba[:, :, 3][upper_cloth_mask > 0] = 0
        # ----------------------------------------------------------------

        if garment_type == 'pants':
            # --- HIGH-QUALITY LIP ARTIFACT CLEANUP ---
            # DESIGNED ONLY FOR PANTS TO REMOVE WARPING ARTIFACTS
            # 1. Create a "Rough Composite" for LIP to evaluate
            temp_base = person_clean.astype(np.float32)
            temp_overlay = warped_rgba[:, :, :3].astype(np.float32)
            temp_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0
            temp_alpha_3 = np.stack([temp_alpha]*3, axis=2)
            rough_composite = (temp_overlay * temp_alpha_3 + temp_base * (1 - temp_alpha_3)).astype(np.uint8)
            
            # Save "Before" image for reference
            pre_cleanup_path = output_path.replace(".png", "_BEFORE_REMOVAL.png")
            cv2.imwrite(pre_cleanup_path, rough_composite)
            print(f"📸 Saved Pre-Cleanup result to: {pre_cleanup_path}")

            # 2. Run Smooth LIP Human Parsing
            lip_seg_map = self.get_person_mask_lip(rough_composite)
            
            if lip_seg_map is not None:
                # Extract Background (Class 0)
                # Create a smooth binary person mask (Class != 0)
                person_mask_raw = (lip_seg_map != 0).astype(np.uint8) * 255
                
                # Anti-Aliasing: Smoothing the "zig-zag" edges of the binary mask
                person_mask_smooth = cv2.GaussianBlur(person_mask_raw, (5, 5), 0)
                _, person_mask_smooth = cv2.threshold(person_mask_smooth, 127, 255, cv2.THRESH_BINARY)
                
                # --- USER REQUESTED FIX: EXPAND MASK TO PREVENT CUTTING PANTS ---
                # "edges of pant should not remove and the hole from the inside"
                # 1. Fill holes (Closing)
                k_close = np.ones((15, 15), np.uint8)
                person_mask_smooth = cv2.morphologyEx(person_mask_smooth, cv2.MORPH_CLOSE, k_close)

                # 1.5 FILTER NOISE (Small disconnected pieces)
                # "if small segmentation pieces there please do not segment... they will be removed"
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(person_mask_smooth, connectivity=8)
                if num_labels > 1:
                    # Sort components by area (descending), excluding background
                    sizes = stats[1:, cv2.CC_STAT_AREA]
                    if len(sizes) > 0:
                        max_area = np.max(sizes)
                        min_valid_area = max(500, max_area * 0.10) # 10% of largest or 500px min
                        
                        filtered_mask = np.zeros_like(person_mask_smooth)
                        for i in range(1, num_labels):
                            if stats[i, cv2.CC_STAT_AREA] >= min_valid_area:
                                filtered_mask[labels == i] = 255
                        
                        person_mask_smooth = filtered_mask

                # 2. Expand mask (Dilate) to keep flares/pockets that LIP misses
                k_dilate = np.ones((20, 20), np.uint8) # Generous expansion
                person_mask_smooth = cv2.dilate(person_mask_smooth, k_dilate, iterations=1)
                # ----------------------------------------------------------------
                
                # Save LIP Mask for verification
                mask_out_path = output_path.replace(".png", "_LIP_MASK.png")
                cv2.imwrite(mask_out_path, person_mask_smooth)
                print(f"✅ Saved Smooth LIP Mask to: {mask_out_path}")
                
                # 3. Apply Global LIP Masking (Global removal)
                # This follows the LIP model strictly.
                warped_rgba[:, :, 3] = cv2.bitwise_and(warped_rgba[:, :, 3], person_mask_smooth)
        else:
            print(f"⏭️ Skipping LIP Cleanup for {garment_type} mode.")
            
            # 4. Final smoothing of the edges
            final_alpha = warped_rgba[:, :, 3].astype(np.float32)
            final_alpha = cv2.GaussianBlur(final_alpha, (3, 3), 0)
            
            # --- [DISABLED per user request: Hand Trimming Removed] ---
            # mp_mask = self.masker.get_hands_only_mask(original_img if original_img is not None else person)
            # print("👐 Trimming cloth from Hands (Precise)...")
            # inv_mp_mask = cv2.bitwise_not(mp_mask)
            # final_alpha = cv2.bitwise_and(final_alpha.astype(np.uint8), inv_mp_mask)
            
            warped_rgba[:, :, 3] = final_alpha.astype(np.uint8)

        # --- UNIFIED MASK GENERATION (USER REQUESTED) ---
        print("🎭 Generating Unified VTON Mask...")
        # Get final combined mask: Diff (Blurred) + Extremities + New Cloth
        # Note: 'person_clean' variable here holds the input image to the warper (which might be the cleaned one from helper)
        # We pass 'original_img' (true original) and 'clean_img' (explicit clean if passed, else None)
        # If called from Helper: person_clean=CleanTemp, original_img=Original. Diff works.
        # If called from Standalone: person_clean=Original, original_img=Original. Diff is 0. Base mask used.
        
        # If clean_img not explicitly passed, can we infer it?
        # If person_clean is different from original_img, use person_clean as clean.
        target_clean = clean_img if clean_img is not None else person_clean
        target_original = original_img if original_img is not None else person
        
        unified_mask = self.masker.get_final_mask(target_original, warped_rgba[:, :, 3], mode='pants', clean_img=target_clean, garment_type=masker_g_type)
        
        # --- ARTIFICIAL SKIN HELPER (Integrated) ---
        print("  🦴 Drawing Artificial Skin Skeleton...")
        # Define output directory for skin helper
        out_dir = Path(output_path).parent
        
        # USER REQUEST: Only include legs for shorts and skirt.
        should_include_legs = (garment_type in ['shorts', 'skirt'])
        
        # Draw skin ON TOP OF person_clean BEFORE compositing pants
        skin_result, skin_mask = self.skin_helper.process(
            person_clean.copy(), warped_rgba[:, :, 3], out_dir, 
            pose_kps=p_pose, include_arms=False, include_legs=should_include_legs,
            sampling_img=target_original, upper_body_mask=upper_cloth_mask
        )

        # --- COMPOSITE PANTS ON TOP OF SKIN ---
        overlay = warped_rgba[:, :, :3].astype(np.float32)
        alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0
        alpha_3 = np.stack([alpha]*3, axis=2)
        # Use skin_result as base
        result = (overlay * alpha_3 + skin_result.astype(np.float32) * (1 - alpha_3)).astype(np.uint8)

        # USER REQUEST: Expand artificial leg mask ONLY in the final mask
        print("  🎭 Expanding Artificial Skin Mask (Legs) for FINAL_MASK...")
        k_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        expanded_skin_mask = cv2.dilate(skin_mask, k_skin, iterations=1)
        
        # Add expanded artificial skin mask to the unified mask
        unified_mask = cv2.bitwise_or(unified_mask, expanded_skin_mask)

        # --- FINAL FACE PROTECTION (Override) ---
        print("  🛡️ Applying Final Face Shield...")
        # Use target_original for face detection
        face_shield = self.masker.get_face_shield(target_original) 
        unified_mask[face_shield > 0] = 0
        
        # Save for User
        unified_mask_path = str(Path(output_path).parent / "final_mask.png")
        cv2.imwrite(unified_mask_path, unified_mask)
        print(f"✅ Saved Unified Mask to: {unified_mask_path}")

        print("Compositing Final Result...")
        cv2.imwrite(output_path, result)
        print(f"Saved to {output_path} (Type: {garment_type})")
        
        
        # 5. Debug Skeleton (Removed as per user request)
        # out_skel = self.draw_skeleton(out, p_pose, color=(0, 255, 0), thickness=4) 
        # skel_path = output_path.replace(".png", "_SKELETON.png")
        # cv2.imwrite(skel_path, out_skel)

def POSE_INT(pt):
    return np.array([int(pt[0]), int(pt[1])])

def main():
    # Discover project root (4 levels up from Modules/Virtual_Tryon2)
    base_dir = Path(__file__).resolve().parents[3]
    
    # Standardized paths from models_downloader.py
    SEG = base_dir / "models" / "SegFormerB2Clothes" / "segformer_b2_clothes.onnx"
    B3 = base_dir / "models" / "segformer-b3-fashion"
    
    # Check if models exist
    if not SEG.exists():
        print(f"❌ Error: B2 model not found at {SEG}")
        return
    
    warper = LBSPantsWarper(SEG, B3)
    
    # Example paths
    PERSON = r"person.png"
    CLOTH = r"cloth.png"
    OUT = r"result.png"
    
    warper.process(PERSON, CLOTH, OUT)

if __name__ == "__main__":
    main()
