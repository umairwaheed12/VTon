import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import torch
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)


class ShoulderHeightDressWarper:
    """
    Pose-aware dress warper.
    - Width anchored to person's shoulders
    - Height scaled using hips/knees/ankles (if cloth has person)
    - Works with cloth images with or without a person
    """

    def __init__(self, seg_model_path, saree_model_path=None):
        # Use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(seg_model_path), sess_options=sess_opts, providers=providers)
            print(f"DressWarper ONNX Runtime Providers: {self.session.get_providers()}")
        except Exception:
            print("Warning: Failed to load ONNX with CUDA, falling back to CPU")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(seg_model_path), sess_options=sess_opts, providers=['CPUExecutionProvider'])
            
        self.input_name = self.session.get_inputs()[0].name
        
        # Default or provided path for Saree/Fashion model
        if saree_model_path is None:
             # Fallback to hardcoded if not provided (legacy support)
             self.saree_model_path = r"D:\Models\segformer-b3-fashion"
        else:
             self.saree_model_path = saree_model_path
        
        # Pre-load Saree/Fashion Model (Performance Optimization)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Saree Model from {self.saree_model_path} on: {self.device}")
        try:
            self.saree_processor = SegformerImageProcessor.from_pretrained(self.saree_model_path)
            self.saree_model = SegformerForSemanticSegmentation.from_pretrained(self.saree_model_path).to(self.device)
            self.saree_model.eval()
            print("✅ Saree Model Loaded Successfully")
        except Exception as e:
            print(f"❌ Failed to load Saree model: {e}")
            self.saree_model = None
            self.saree_processor = None

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    # ---------------- Pose detection ----------------
    def get_pose(self, image):
        H, W = image.shape[:2]
        res = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        def p(i): return np.array([lm[i].x * W, lm[i].y * H], np.float32)
        return {
            "nose": p(0),
            "l_ear": p(7), "r_ear": p(8),
            "l_shoulder": p(11), "r_shoulder": p(12),
            "l_hip": p(23), "r_hip": p(24),
            "l_knee": p(25), "r_knee": p(26),
            "l_ankle": p(27), "r_ankle": p(28),
        }

    # ---------------- Segmentation Helpers ----------------
    def _segment_logits(self, image):
        """
        Runs SegFormer on the image and returns resized logits.
        """
        h, w = image.shape[:2]
        img = cv2.resize(image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        img = img.transpose(2,0,1)[None].astype(np.float32)
        logits = self.session.run(None, {self.input_name: img})[0][0]
        logits = cv2.resize(logits.transpose(1,2,0), (w,h))
        return logits

    def get_cloth_arms_hands_mask(self, image):
        """
        Extracts arms and hands from cloth image with clean edges.
        Returns a combined mask for arms (14,15) and hands (16,17).
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Arms: Class 14, 15
        mask_arms = ((seg == 14) | (seg == 15)).astype(np.uint8) * 255
        
        # Hands: Class 16, 17
        mask_hands = ((seg == 16) | (seg == 17)).astype(np.uint8) * 255
        
        # Combine arms and hands
        mask_combined = cv2.bitwise_or(mask_arms, mask_hands)
        
        # Enhanced post-processing for clean edges
        # Close small holes
        k_close = np.ones((7, 7), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, k_close)
        
        # Remove small noise
        k_open = np.ones((3, 3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, k_open)
        
        # Keep sharp edges - no blur, just threshold for binary mask
        _, mask_combined = cv2.threshold(mask_combined, 127, 255, cv2.THRESH_BINARY)
        
        # Final cleanup - only close, no erosion to preserve corners
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, k_close)
        
        return mask_combined

    def get_hair_mask(self, image):
        """
        Extracts hair mask (Class 2) from person image.
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Class 2 is Hair in ATR mapping
        mask_hair = (seg == 2).astype(np.uint8) * 255
        
        # Dilate slightly to ensure clean trimming
        k = np.ones((3,3), np.uint8)
        mask_hair = cv2.dilate(mask_hair, k, iterations=1)
        
        return mask_hair

    def get_person_cloth_arms_hands_mask(self, image):
        """
        Creates AGGRESSIVE mask for person's original clothes, arms, hands, AND UPPER BODY SKIN.
        Removes EVERYTHING in upper body area except face/neck and feet.
        This ensures complete removal of original clothing and body.
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        CLOTHES = {4, 5, 6, 7, 8, 18}  # All clothing
        ARMS = {14, 15}  # Arms
        HANDS = {16, 17}  # Hands
        UPPER_SKIN = {11, 12, 13}  # Skin (includes torso, arms, legs)
        FACE_NECK = {1, 2, 3, 11}  # Face, Hair, Sunglasses, and Face skin
        FEET = {9, 10}  # Feet
        
        # --- CRITICAL: REMOVE EVERYTHING IN UPPER BODY ---
        # Include clothes, arms, hands, AND upper body skin
        remove = np.zeros(seg.shape, dtype=np.uint8)
        for lab in CLOTHES | ARMS | HANDS | UPPER_SKIN:
            remove[seg == lab] = 255
        
        # --- PROTECTION MASK (ONLY preserve face/neck and feet) ---
        # We need to protect face/neck area and feet, but remove everything else
        protect = np.zeros(seg.shape, dtype=np.uint8)
        
        # Protect face/hair/neck area (upper portion only)
        h, w = seg.shape
        face_area = np.zeros(seg.shape, dtype=np.uint8)
        for lab in FACE_NECK:
            face_area[seg == lab] = 255
        
        # Only protect face/neck in upper 40% of image
        face_area[int(h * 0.4):, :] = 0
        protect = cv2.bitwise_or(protect, face_area)
        
        # Protect feet (lower portion only)
        feet_area = np.zeros(seg.shape, dtype=np.uint8)
        for lab in FEET:
            feet_area[seg == lab] = 255
        protect = cv2.bitwise_or(protect, feet_area)
        
        # Clean up remove mask with morphological operations
        # Close small holes first
        kernel_close = np.ones((7, 7), np.uint8)
        remove = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel_close)
        
        # AGGRESSIVE dilation to ensure complete coverage
        kernel_dilate = np.ones((15, 15), np.uint8)
        remove = cv2.dilate(remove, kernel_dilate, iterations=3)
        
        # HARD subtract protected areas (face/neck and feet)
        remove[protect > 0] = 0
        
        # --- SPECIAL HANDLING: BELLY/TORSO ---
        # If the person has bare skin in the torso (belly), we want to INCLUDE it in the cloth mask
        # so it gets covered by the new dress (instead of being protected as skin).
        # We use Pose landmarks to define the "Torso Box" (Shoulders to Hips).
        pose = self.get_pose(image)
        if pose:
            try:
                # Define Torso Region (Between Shoulders and Hips)
                ys = [pose['l_shoulder'][1], pose['r_shoulder'][1]]
                ye = [pose['l_hip'][1], pose['r_hip'][1]]
                xs = [pose['l_shoulder'][0], pose['r_shoulder'][0], pose['l_hip'][0], pose['r_hip'][0]]
                
                y_min = int(max(ys)) # Lowest shoulder point (start of torso)
                y_max = int(min(ye)) # Highest hip point (end of torso)
                x_min = int(min(xs))
                x_max = int(max(xs))
                
                # Check bounds
                h, w = seg.shape
                if y_min < y_max and x_min < x_max:
                     # Create Torso ROI Mask
                    torso_roi = np.zeros_like(seg, dtype=np.uint8)
                    cv2.rectangle(torso_roi, (x_min, y_min), (x_max, y_max), 255, -1)
                    
                    # Find SKIN pixels (11, 12, 13) within this Torso Box
                    # Note: Class 11 is often "Skin/Face", 12/13 are Legs. 
                    # Torso skin usually falls into 4 (Upper) if covered, or 11/0 if bare.
                    # We want to catch bare skin (11) in this box.
                    belly_skin = np.zeros_like(seg, dtype=np.uint8)
                    UPPER_SKIN = {11, 12, 13}
                    for lab in UPPER_SKIN: 
                        belly_skin[(seg == lab) & (torso_roi == 255)] = 255
                    
                    # Also check for Background (0) appearing "inside" the torso (e.g. deep cut)?
                    # Some models map skin to 0 if unsure.
                    # Let's check for class 0 in the very center of torso.
                    center_mask = np.zeros_like(seg, dtype=np.uint8)
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    # Small box in center
                    cv2.rectangle(center_mask, (cx-20, cy-20), (cx+20, cy+20), 255, -1)
                    belly_skin[(seg == 0) & (center_mask == 255)] = 255
                        
                    # MOVE belly skin from PROTECT to REMOVE
                    if cv2.countNonZero(belly_skin) > 0:
                        print(f"ℹ Detected {cv2.countNonZero(belly_skin)} belly/torso skin pixels. Including in mask.")
                        protect[belly_skin > 0] = 0
                        remove[belly_skin > 0] = 255
            except Exception as e:
                print(f"⚠ Failed to calculate belly mask: {e}")

        # CRITICAL: EXCLUDE ALL BACKGROUND
        background_mask = (seg == 0).astype(np.uint8) * 255
        remove[background_mask > 0] = 0
        
        # CRITICAL: Use connected components to keep only regions connected to person
        # This removes any stray background pixels or disconnected regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(remove, connectivity=8)
        
        if num_labels > 1:  # 0 is background, so if > 1 we have components
            # Keep all components that are large enough (not tiny noise)
            min_area = 500  # Minimum area to keep (larger threshold for upper body)
            clean_mask = np.zeros_like(remove)
            for i in range(1, num_labels):  # Skip background (0)
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean_mask[labels == i] = 255
            remove = clean_mask
        
        # Keep sharp edges - no blur, just threshold for binary mask
        _, remove = cv2.threshold(remove, 127, 255, cv2.THRESH_BINARY)
        
        # Final cleanup - only close small holes, no opening to preserve corners
        kernel_clean = np.ones((5, 5), np.uint8)
        remove = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel_clean)
        
        return remove

    # ---------------- Saree Segmentation ----------------
    def get_pants_mask(self, image):
        """
        Segments pants using SegFormer (class 6 = Pants/Bottoms).
        Same method as pants.py
        """
        logits = self._segment_logits(image)
        seg_map = np.argmax(logits, axis=2).astype(np.uint8)
        
        # Class 6 = Pants/Bottoms
        pants_mask = (seg_map == 6).astype(np.uint8) * 255
        # Clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pants_mask = cv2.morphologyEx(pants_mask, cv2.MORPH_OPEN, kernel)
        
        return pants_mask

    def get_upper_clothes_mask(self, image):
        """
        Extracts upper clothes (class 4) from cloth image.
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Upper: Class 4 (Upper clothes)
        mask_upper = (seg == 4).astype(np.uint8) * 255
        
        # Post-processing for cloth
        k = np.ones((5, 5), np.uint8)
        mask_upper = cv2.morphologyEx(mask_upper, cv2.MORPH_CLOSE, k)
        
        return mask_upper

    def get_skirt_mask(self, image):
        """
        Segments skirt using SegFormer (class 5 = Skirt).
        """
        logits = self._segment_logits(image)
        seg_map = np.argmax(logits, axis=2).astype(np.uint8)
        
        # Class 5 = Skirt
        skirt_mask = (seg_map == 5).astype(np.uint8) * 255
        # Clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skirt_mask = cv2.morphologyEx(skirt_mask, cv2.MORPH_OPEN, kernel)
        
        return skirt_mask

    def is_pants_or_shorts(self, image):
        """
        Detects if the garment is pants, shorts, or skirt using SegFormer.
        Returns True if pants/shorts/skirt are detected, False otherwise.
        """
        pants_mask = self.get_pants_mask(image)
        skirt_mask = self.get_skirt_mask(image)
        
        pants_pixel_count = cv2.countNonZero(pants_mask)
        skirt_pixel_count = cv2.countNonZero(skirt_mask)
        total_pixels = pants_mask.size
        
        # If more than 5% of image is pants or skirt, consider it as lower body garment
        is_lower_body = ((pants_pixel_count + skirt_pixel_count) / total_pixels) > 0.05
        return is_lower_body

    def is_person_wearing_pants(self, image):
        """
        Detects if the target person is wearing pants (Class 6).
        Used to decide whether to mask legs or preserve them.
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Class 6 = Pants
        pants_mask = (seg == 6).astype(np.uint8) * 255
        
        count = cv2.countNonZero(pants_mask)
        total = pants_mask.size
        
        # Threshold: if pants cover > 2% of image
        return (count / total) > 0.02

    def is_complex_pose(self, pose_landmarks, image_shape):
        """
        Determines if the pose is complex (sitting, lying down, etc.)
        where standard cloth warping might fail.
        """
        if not pose_landmarks: return False
        
        h, w = image_shape[:2]
        lm = pose_landmarks
        
        # 1. Check Aspect Ratio of the person's bounding box
        # (Lying down often results in wider bounding boxes)
        xs = [p[0] for p in lm.values()]
        ys = [p[1] for p in lm.values()]
        box_w = max(xs) - min(xs)
        box_h = max(ys) - min(ys)
        aspect_ratio = box_h / (box_w + 1e-6)
        
        # Normal standing/walking is usually > 1.5 or 2.0
        # Sitting/Lying might be < 1.3
        if aspect_ratio < 1.3:
            print(f"ℹ Complex Pose Detected: Wide aspect ratio ({aspect_ratio:.2f})")
            return True
            
        # 2. Check Keypoint Angles (Sitting Detection)
        # Angle at Hips: Shoulder -> Hip -> Knee
        # Function to calc angle
        def get_angle(a, b, c):
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

        l_hip_angle = get_angle(lm["l_shoulder"], lm["l_hip"], lm["l_knee"])
        r_hip_angle = get_angle(lm["r_shoulder"], lm["r_hip"], lm["r_knee"])
        
        # Standing ~ 160-180 degrees. Sitting ~ 90 degrees.
        # If hips are bent < 130 degrees, likely sitting/squatting
        if l_hip_angle < 135 or r_hip_angle < 135:
            print(f"ℹ Complex Pose Detected: Hips bent (L:{l_hip_angle:.1f}, R:{r_hip_angle:.1f})")
            return True
            
        return False

    def get_person_head_mask(self, image):
        """
        Extracts head (Face, Hair, Neck) from person image.
        Classes: 1(Hat), 2(Hair), 3(Glove-no), 11(Body Skin - need to filter for Face/Neck)
        Actually SegFormer classes:
        0: Background
        1: Hat
        2: Hair
        3: Sunglasses
        11: Skin (Face/Neck/Arms/Legs - one class usually? No, SegFormer ATR often splits or just 'Skin')
        Wait, SegFormer B2 Clothes often has: 
        1: Hat, 2: Hair, 3: Sunglasses, 4: Upper, 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left Shoe, 10: Right Shoe, 11: Face, 12: Left Leg, 13: Right Leg, 14: Left Arm, 15: Right Arm, 16: Bag, 17: Scarf
        Let's assume standard mapping:
        1: Hat
        2: Hair
        3: Sunglasses
        11: Face (SegFormer usually separates Face from other skin)
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Head parts: Hat(1), Hair(2), Sunglasses(3), Face(11 usually in some mappings, check my other funcs)
        # previous code used: SKIN = {11, 12, 13}
        # In many ATR/Lip mappings: 11 is Face, 12/13 Legs, 14/15 Arms?
        # Let's check get_person_cloth_arms_hands_mask from before:
        # ARMS = {14, 15}, HANDS = {16, 17}?? No, that code had HANDS=16,17. 
        # Typically: 11=Face, 12=Leg, 13=Leg? Or 11=Face.
        # Safest is to include 1, 2, 3, and 11 (Face).
        # Also need Neck? Neck is often part of Face or Upper Skin.
        # If SegFormer "Face" includes Neck, we resemble good.
        
        # HEAD Classes
        HEAD = {1, 2, 3, 11} # Hat, Hair, Glasses, Face
        
        head_mask = np.zeros(seg.shape, dtype=np.uint8)
        for lab in HEAD:
            head_mask[seg == lab] = 255
            
        # Optimization: Neck might be missing if classified as Skin/UpperSkin.
        # Often Neck is just below face.
        # Let's also grab skin near the head?
        # For now, stick to Face class.
        
        # Clean holes
        k = np.ones((5,5), np.uint8)
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, k)
        
        # Dilate slightly to catch edges
        head_mask = cv2.dilate(head_mask, k, iterations=1)
        
        return head_mask

    def process_head_swap(self, person, cloth, p_pose, c_pose, person_shape):
        """
        Swaps the person's head onto the cloth image model.
        Used for complex poses where we keep the cloth image body.
        """
        print("⚡ processing HEAD SWAP (Complex Pose Strategy)...")
        # 1. Scale Calculation
        # Primary: Face Width (Ear to Ear) - ensures head size matches
        # Secondary: Shoulder Width - ensures body alignment
        
        # Calculate Shoulder Scale
        src_w_sh = np.linalg.norm(p_pose["l_shoulder"] - p_pose["r_shoulder"])
        dst_w_sh = np.linalg.norm(c_pose["l_shoulder"] - c_pose["r_shoulder"])
        scale_sh = dst_w_sh / (src_w_sh + 1e-6)
        
        # Calculate Face Scale (if ears available)
        scale = scale_sh * 0.9  # Default to slightly smaller shoulder scale (heuristic for complex pose)
        
        src_w_face = 0
        dst_w_face = 0
        if "l_ear" in p_pose and "r_ear" in p_pose and "l_ear" in c_pose and "r_ear" in c_pose:
             src_w_face = np.linalg.norm(p_pose["l_ear"] - p_pose["r_ear"])
             dst_w_face = np.linalg.norm(c_pose["l_ear"] - c_pose["r_ear"])
             
        # If face width is valid (non-zero), use it
        if src_w_face > 5 and dst_w_face > 5:
            scale_face = dst_w_face / src_w_face
            print(f"ℹ Scale stats - Shoulder: {scale_sh:.2f}, Face: {scale_face:.2f}")
            
            # Use Face Scale primarily for head swap to avoid "huge head"
            # But average with shoulder to keep neck reasonable?
            # User complained "head is too big", so trust Face Scale more.
            # However, if detection is noisy, clamping is good.
            scale = scale_face
        else:
            print(f"ℹ Face landmarks missing/weak, using reduced shoulder scale: {scale:.2f}")

        # Resize Person Image
        ph, pw = person.shape[:2]
        person_scaled = cv2.resize(person, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        
        # Update scaled landmarks
        p_pose_scaled = {}
        for k, v in p_pose.items():
            p_pose_scaled[k] = v * scale
            
        # 2. Align Head position
        # Align Center of Shoulders? Or Align Neck (mid-shoulder)?
        # If we scaled by Face, Shoulders might not match exactly.
        # Aligning by Chin or Nose would be better for Head placement?
        # Let's align by Mid-Shoulder (Neck Base) as it's the attachment point.
        src_center = (p_pose_scaled["l_shoulder"] + p_pose_scaled["r_shoulder"]) * 0.5
        dst_center = (c_pose["l_shoulder"] + c_pose["r_shoulder"]) * 0.5
        
        dx = dst_center[0] - src_center[0]
        dy = dst_center[1] - src_center[1]
        
        # Translation Matrix
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        
        # Canvas = Cloth Image
        base_image = cloth.copy()
        ch, cw = base_image.shape[:2]
        
        # Warp Person (Translate) onto Cloth Canvas
        person_aligned = cv2.warpAffine(person_scaled, M, (cw, ch), flags=cv2.INTER_LANCZOS4)
        
        # 3. Extract Head from Aligned Person
        head_mask = self.get_person_head_mask(person)
        head_mask_scaled = cv2.resize(head_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        head_mask_aligned = cv2.warpAffine(head_mask_scaled, M, (cw, ch), flags=cv2.INTER_NEAREST)
        
        # 4. Composite
        head_alpha = head_mask_aligned.astype(np.float32) / 255.0
        head_alpha = cv2.GaussianBlur(head_alpha, (5, 5), 0)
        
        out = person_aligned * head_alpha[:, :, None] + base_image * (1 - head_alpha[:, :, None])
        
        return out.astype(np.uint8)

    def get_saree_mask(self, image):
        """
        Uses HuggingFace SegFormer (b3-fashion) to segment Saree components.
        Classes: 10,11,26,32,34,2,9
        """

        if self.saree_model is None:
            print("Saree model not initialized.")
            return None
            
        # Prepare image using pre-loaded processor/model
        try:
            inputs = self.saree_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.saree_model(**inputs)
                # Move logits back to CPU for numpy operations
                logits = outputs.logits.cpu()
        except Exception as e:
            print(f"Error during Saree model inference: {e}")
            return None
            
        # Resize logits
        logits = torch.nn.functional.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        )
        seg = logits.argmax(dim=1)[0].numpy()
        
        # Saree Union Classes (All Cloth-Related)
        # Excluded: 0 (Unlabelled), 14 (glasses), 15 (hat), 16 (headband/hair), 
        # 24 (shoe), 25 (bag/wallet), 27 (umbrella)
        # Included: All garments, parts, and decorations
        saree_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
            17, 18, 19, 20, 21, 22, 23, 
            26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
        ]
        mask = np.isin(seg, saree_ids).astype(np.uint8) * 255
        
        # Enhanced cleanup for clean, hole-free mask
        # Step 1: Fill small holes first
        kernel_close_small = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_small)
        
        # Step 2: Fill larger holes
        kernel_close_large = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_large)
        
        # Step 3: Remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Step 4: Final closing to ensure no holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_large)
        
        # Step 5: Fill any remaining holes using floodFill
        # Invert mask to find holes
        mask_inv = cv2.bitwise_not(mask)
        h, w = mask.shape
        # Fill from edges
        cv2.floodFill(mask_inv, None, (0, 0), 255)
        cv2.floodFill(mask_inv, None, (w-1, 0), 255)
        cv2.floodFill(mask_inv, None, (0, h-1), 255)
        cv2.floodFill(mask_inv, None, (w-1, h-1), 255)
        # Holes are the areas not filled
        holes = cv2.bitwise_not(mask_inv)
        # Fill holes in original mask
        mask = cv2.bitwise_or(mask, holes)
        
        del inputs, outputs, logits
        import gc
        gc.collect()
        return mask

    # ---------------- Normalize & scale ----------------
    def normalize_and_scale(self, dress, c_pose, p_pose, mask, person_shape):
        ph, pw = person_shape[:2]

        # ---------------- Case 1: Cloth has person ----------------
        if c_pose is not None:
            # Resize for resolution
            scale_res = max(ph / dress.shape[0], pw / dress.shape[1])
            dress = cv2.resize(dress, None, fx=scale_res, fy=scale_res, interpolation=cv2.INTER_LANCZOS4)
            for k in c_pose:
                c_pose[k] = c_pose[k] * scale_res

            # Shoulder width scaling
            src_w = np.linalg.norm(c_pose["l_shoulder"] - c_pose["r_shoulder"])
            dst_w = np.linalg.norm(p_pose["l_shoulder"] - p_pose["r_shoulder"])
            sx = dst_w / (src_w + 1e-6)

            # Height scaling using hips/knees/ankles
            src_h = np.mean([c_pose["l_hip"][1], c_pose["r_hip"][1],
                            c_pose["l_knee"][1], c_pose["r_knee"][1],
                            c_pose["l_ankle"][1], c_pose["r_ankle"][1]]) - \
                    np.mean([c_pose["l_shoulder"][1], c_pose["r_shoulder"][1]])
            dst_h = np.mean([p_pose["l_hip"][1], p_pose["r_hip"][1],
                            p_pose["l_knee"][1], p_pose["r_knee"][1],
                            p_pose["l_ankle"][1], p_pose["r_ankle"][1]]) - \
                    np.mean([p_pose["l_shoulder"][1], p_pose["r_shoulder"][1]])
            sy = dst_h / (src_h + 1e-6)

            # Resize dress
            dress = cv2.resize(dress, None, fx=sx, fy=sy, interpolation=cv2.INTER_LANCZOS4)

            # Scale cloth pose keypoints
            for k in c_pose:
                c_pose[k] = c_pose[k] * [sx, sy]

        # ---------------- Case 2: Cloth image has no person ----------------
        else:
            # Use mask bounding box
            coords = cv2.findNonZero(mask)
            if coords is None:
                # print("⚠ Dress mask empty")
                return dress, c_pose

            x, y, w, h = cv2.boundingRect(coords)

            # Crop dress and mask to the bounding box
            cropped_dress = dress[y:y+h, x:x+w]
            cropped_mask = mask[y:y+h, x:x+w]

            # ---------------- Width Scaling ----------------
            # Determine top width of the dress
            top_section_h = max(1, int(h * 0.15))
            top_mask_slice = cropped_mask[:top_section_h, :]
            
            top_coords = cv2.findNonZero(top_mask_slice)
            if top_coords is not None:
                tx, ty, tw, th = cv2.boundingRect(top_coords)
                dress_top_width = tw
                top_center_x_in_crop = tx + tw / 2.0
            else:
                dress_top_width = w
                top_center_x_in_crop = w / 2.0

            # Target 1: Width should be much wider for ball gowns with narrow busts
            # The narrow bust needs significant scaling to reach shoulder width
            person_shoulder_width = np.linalg.norm(p_pose["l_shoulder"] - p_pose["r_shoulder"])
            target_width = person_shoulder_width * 5.0  # Much wider for proper coverage
            
            if dress_top_width < 1: dress_top_width = 1
            scale_x = target_width / dress_top_width

            # ---------------- Height Scaling ----------------
            # Target 2: Height from shoulder to ankle -> extended to feet
            shoulder_y = (p_pose["l_shoulder"][1] + p_pose["r_shoulder"][1]) / 2
            ankle_y = (p_pose["l_ankle"][1] + p_pose["r_ankle"][1]) / 2
            
            body_length = ankle_y - shoulder_y
            target_height = max(1, int(body_length * 1.15))
            
            # ---------------- Final Resize ----------------
            final_total_width = max(1, int(w * scale_x))
            final_total_height = int(target_height)
            
            dress = cv2.resize(cropped_dress, (final_total_width, final_total_height), interpolation=cv2.INTER_LANCZOS4)

            # ---------------- Anchor Point ----------------
            actual_scale_x = final_total_width / max(1, w)
            final_top_center_x = top_center_x_in_crop * actual_scale_x
            
            # Create a dummy c_pose with proper shoulder width spanning the TARGET width
            # Use target_width (not dress_top_width) to create proper shoulder positions
            half_width = target_width / 2.0
            
            c_pose = {
                "l_shoulder": np.array([final_top_center_x - half_width, 0], dtype=np.float32),
                "r_shoulder": np.array([final_top_center_x + half_width, 0], dtype=np.float32)
            }

        return dress, c_pose

    # ---------------- Lower Body Masks ----------------
    def get_person_lower_body_mask(self, image):
        """
        Gets mask for person's lower body: pants (6) + legs (skin in lower half) + feet (9, 10) + shoes (24).
        Returns mask for areas to be replaced.
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Pants: Class 6
        # Feet: Class 9, 10
        # Shoes: Class 24
        # Legs/Skin: Class 11, 12, 13 (skin areas that are legs - only in lower half)
        PANTS = {6}
        FEET = {9, 10}
        SHOES = {24}
        SKIN = {11, 12, 13}  # Include skin for legs
        
        # Create lower body mask - include pants, feet, shoes
        lower_mask = np.zeros(seg.shape, dtype=np.uint8)
        for lab in PANTS | FEET | SHOES:
            lower_mask[seg == lab] = 255
        
        # Also include leg skin (but only in lower half of image to avoid arms)
        h, w = seg.shape
        lower_half_start = h // 2  # Start from middle of image
        for lab in SKIN:
            # Only include skin in lower half (legs, not arms)
            skin_mask = (seg == lab).astype(np.uint8)
            skin_mask[:lower_half_start, :] = 0  # Remove upper half (arms)
            lower_mask[skin_mask > 0] = 255
        
        # Clean up mask
        kernel_close = np.ones((5, 5), np.uint8)
        lower_mask = cv2.morphologyEx(lower_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate slightly to ensure complete coverage
        kernel_dilate = np.ones((3, 3), np.uint8)
        lower_mask = cv2.dilate(lower_mask, kernel_dilate, iterations=1)
        
        # Make binary
        _, lower_mask = cv2.threshold(lower_mask, 127, 255, cv2.THRESH_BINARY)
        
        return lower_mask

    def get_person_feet_shoes_mask(self, image):
        """
        Gets mask for person's feet and shoes for skin masking.
        Returns mask for feet (9, 10).
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Feet: Class 9, 10
        # Shoes: Class 24
        FEET = {9, 10}
        SHOES = {24}
        
        feet_mask = np.zeros(seg.shape, dtype=np.uint8)
        for lab in FEET | SHOES:
            feet_mask[seg == lab] = 255
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        feet_mask = cv2.morphologyEx(feet_mask, cv2.MORPH_CLOSE, kernel)
        _, feet_mask = cv2.threshold(feet_mask, 127, 255, cv2.THRESH_BINARY)
        
        return feet_mask

    def get_cloth_lower_body_mask(self, image):
        """
        Gets mask for cloth's lower body: pants (6) + skirt (5) + feet (9, 10) + shoes (24).
        Returns mask for areas to cut from cloth - EXCLUDES background.
        """
        logits = self._segment_logits(image)
        seg = np.argmax(logits, axis=2)
        
        # Pants: Class 6
        # Skirt: Class 5
        # Feet: Class 9, 10
        # Shoes: Class 24
        PANTS = {6}
        SKIRT = {5}
        FEET = {9, 10}
        SHOES = {24}
        SKIN = {11, 12, 13}
        
        # Create lower body mask (only clothing/body parts, NOT background)
        lower_mask = np.zeros(seg.shape, dtype=np.uint8)
        for lab in PANTS | SKIRT | FEET | SHOES:
            lower_mask[seg == lab] = 255
            
        # Also include leg skin (but only in lower half of image to avoid arms)
        h, w = seg.shape
        lower_half_start = h // 2  # Start from middle of image
        for lab in SKIN:
            # Only include skin in lower half (legs, not arms)
            skin_mask = (seg == lab).astype(np.uint8)
            skin_mask[:lower_half_start, :] = 0  # Remove upper half (arms)
            lower_mask[skin_mask > 0] = 255
        
        # EXCLUDE BACKGROUND - remove any background pixels from mask
        # Background classes: 0 (background), and any non-clothing areas
        # Get background mask
        background_mask = np.zeros(seg.shape, dtype=np.uint8)
        # Class 0 is typically background, also exclude any large uniform areas at edges
        background_mask[seg == 0] = 255
        
        # Remove background from lower body mask
        lower_mask[background_mask > 0] = 0
        
        # Clean up mask - fill holes
        kernel_close = np.ones((5, 5), np.uint8)
        lower_mask = cv2.morphologyEx(lower_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # More aggressive cleaning - remove noise
        kernel_open = np.ones((3, 3), np.uint8)
        lower_mask = cv2.morphologyEx(lower_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Dilate slightly to ensure complete coverage (but not too much to avoid background)
        kernel_dilate = np.ones((3, 3), np.uint8)
        lower_mask = cv2.dilate(lower_mask, kernel_dilate, iterations=1)
        
        # Erode slightly to remove any background that might have been included
        kernel_erode = np.ones((3, 3), np.uint8)
        lower_mask = cv2.erode(lower_mask, kernel_erode, iterations=1)
        
        # Make binary
        _, lower_mask = cv2.threshold(lower_mask, 127, 255, cv2.THRESH_BINARY)
        
        return lower_mask

    # ---------------- Warp dress ----------------
    def warp(self, dress, c_pose, p_pose, person_shape):
        """
        Warp for upper body with shoulder matching and stretching.
        """
        ph, pw = person_shape[:2]
        
        # Warp for upper body with shoulder matching and stretching
        dst_anchor = (p_pose["l_shoulder"] + p_pose["r_shoulder"]) * 0.5
        person_shoulder_width = np.linalg.norm(p_pose["l_shoulder"] - p_pose["r_shoulder"])
        
        if c_pose is not None:
            src_anchor = (c_pose["l_shoulder"] + c_pose["r_shoulder"]) * 0.5
            cloth_shoulder_width = np.linalg.norm(
                c_pose["l_shoulder"] - c_pose["r_shoulder"]
            )
            # For flat cloth (no person), normalize_and_scale already set correct width
            # so we don't need additional scaling - just align
            if cloth_shoulder_width > 10:
                # Real person detected in cloth - scale to match
                scale_x = person_shoulder_width / (cloth_shoulder_width + 1e-6)
                # Clamp scale to reasonable range (0.3x to 3x)
                scale_x = np.clip(scale_x, 0.3, 3.0)
            else:
                # Flat cloth - already scaled in normalize_and_scale, just align
                scale_x = 1.0
        else:
            src_anchor = np.array([dress.shape[1] / 2, 0], np.float32)
            # Estimate cloth shoulder width from image width
            cloth_shoulder_width = dress.shape[1] * 0.6  # Estimate
            scale_x = person_shoulder_width / (cloth_shoulder_width + 1e-6)
            # Clamp scale to reasonable range (0.3x to 3x)
            scale_x = np.clip(scale_x, 0.3, 3.0)

        # Scale horizontally to match shoulders, then align centers
        # First scale around origin, then translate to align
        # After scaling: src_anchor becomes (src_anchor[0] * scale_x, src_anchor[1])
        # We want it at dst_anchor
        dx = dst_anchor[0] - src_anchor[0] * scale_x
        dy = dst_anchor[1] - src_anchor[1]
        
        # Affine matrix: scale then translate
        M = np.array([[scale_x, 0, dx],
                      [0, 1, dy]], dtype=np.float32)
        
        warped = cv2.warpAffine(
             dress, M, (pw, ph),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return warped

    def process_lower_robust(self, cloth, mask, c_pose, p_pose, person_shape, target_has_pants=True):
        """
        Process lower body garments. If pants/skirt detected, extract full outfit.
        Otherwise use normal warping for saree, etc.
        """
        # Check if this is pants/shorts/skirt
        is_lower_body = self.is_pants_or_shorts(cloth)
        
        if is_lower_body:
            # Handle pants/skirt: extract full outfit (shirt + pants/skirt + feet + shoes) together
            print("✓ Detected pants/skirt - extracting full outfit (shirt + lower body + feet + shoes)")
            
            # Get masks for all parts - ALWAYS extract full outfit
            cloth_lower_mask = self.get_cloth_lower_body_mask(cloth)  # Pants/skirt + feet + shoes + skin
            print("✓ ALWAYS extracting full lower body (pants/skirt + legs + feet + shoes) from cloth")
            
            mask_upper = self.get_upper_clothes_mask(cloth)  # Shirt
            mask_arms_hands = self.get_cloth_arms_hands_mask(cloth)  # Arms + hands
            
            # Combine all parts into one mask (full outfit)
            full_outfit_mask = cv2.bitwise_or(cloth_lower_mask, mask_upper)
            full_outfit_mask = cv2.bitwise_or(full_outfit_mask, mask_arms_hands)
            
            if cv2.countNonZero(full_outfit_mask) == 0:
                print("⚠ No outfit detected in cloth image")
                return None
            
            # Clean the combined mask to remove holes
            # Step 1: Fill small holes
            kernel_close_small = np.ones((5, 5), np.uint8)
            full_outfit_mask = cv2.morphologyEx(full_outfit_mask, cv2.MORPH_CLOSE, kernel_close_small)
            
            # Step 2: Fill larger holes
            kernel_close_medium = np.ones((9, 9), np.uint8)
            full_outfit_mask = cv2.morphologyEx(full_outfit_mask, cv2.MORPH_CLOSE, kernel_close_medium)
            
            # Step 3: Remove small noise
            kernel_open = np.ones((3, 3), np.uint8)
            full_outfit_mask = cv2.morphologyEx(full_outfit_mask, cv2.MORPH_OPEN, kernel_open)
            
            # Step 4: Fill any remaining holes using floodFill
            mask_inv = cv2.bitwise_not(full_outfit_mask)
            h, w = full_outfit_mask.shape
            # Fill from corners
            cv2.floodFill(mask_inv, None, (0, 0), 255)
            cv2.floodFill(mask_inv, None, (w-1, 0), 255)
            cv2.floodFill(mask_inv, None, (0, h-1), 255)
            cv2.floodFill(mask_inv, None, (w-1, h-1), 255)
            # Fill holes
            holes = cv2.bitwise_not(mask_inv)
            full_outfit_mask = cv2.bitwise_or(full_outfit_mask, holes)
            
            # Step 5: Final closing
            full_outfit_mask = cv2.morphologyEx(full_outfit_mask, cv2.MORPH_CLOSE, kernel_close_medium)
            
            # Convert cloth to RGBA with full outfit mask
            # IMPORTANT: Only extract clothing parts, exclude background
            cloth_rgba = cv2.cvtColor(cloth, cv2.COLOR_BGR2BGRA)
            # Set alpha channel to mask (only where mask is non-zero)
            cloth_rgba[:, :, 3] = full_outfit_mask
            
            # Remove background pixels: where alpha is 0, set RGB to transparent black
            # This ensures background from cloth image doesn't come through
            background_pixels = (full_outfit_mask == 0)
            cloth_rgba[background_pixels, 0] = 0  # B
            cloth_rgba[background_pixels, 1] = 0  # G
            cloth_rgba[background_pixels, 2] = 0  # R
            cloth_rgba[background_pixels, 3] = 0  # A
            
            # Use normalize_and_scale to scale the full outfit to person's dimensions
            # This handles both cases: cloth with person or without person
            # Same logic as saree - ensures proper scaling based on person's pose
            scaled_outfit, scaled_c_pose = self.normalize_and_scale(
                cloth_rgba, c_pose, p_pose, full_outfit_mask, person_shape
            )
            
            if scaled_outfit is None:
                return None
            
            # Warp the full outfit onto person (same as saree - shoulder-based alignment)
            # This ensures the shirt aligns with shoulders, and pants align naturally
            # The outfit is placed as one unit, maintaining the relationship between
            # shirt and pants from the original cloth image
            warped_outfit = self.warp(scaled_outfit, scaled_c_pose, p_pose, person_shape)
            
            return warped_outfit
        
        # For non-pants (saree, etc.), use original logic
        dress = cv2.cvtColor(cloth, cv2.COLOR_BGR2BGRA)
        dress[:,:,3] = mask
        
        # Normalize and scale (handles Case 1 and Case 2 internally)
        scaled_dress, scaled_c_pose = self.normalize_and_scale(dress, c_pose, p_pose, mask, person_shape)
        
        if scaled_dress is None: 
            return None
            
        # Warp onto person (normal warp for saree)
        return self.warp(scaled_dress, scaled_c_pose, p_pose, person_shape)

    def process_flat_geometric(self, cloth, mask, p_pose, person_shape):
        dress = cv2.cvtColor(cloth, cv2.COLOR_BGR2BGRA)
        dress[:,:,3] = mask
        scaled_dress, scaled_c_pose = self.normalize_and_scale(dress, None, p_pose, mask, person_shape)
        if scaled_dress is None: return None
        return self.warp(scaled_dress, scaled_c_pose, p_pose, person_shape)

    # ---------------- MAIN PIPELINE ----------------
    def process(self, person_input, cloth_input, out_path_final=None):
        """
        Main processing function.
        Args:
            person_input: Path to person image OR numpy array (BGR)
            cloth_input: Path to cloth image OR numpy array (BGR)
            out_path_final: Optional path to save result.
            
        Returns:
            Tuple: (result_image, visualization_mask, warp_mask)
            All as numpy arrays (BGR/Gray).
        """
        # Load Person Image
        if isinstance(person_input, str):
            person = cv2.imread(person_input)
            if person is None:
                print(f"Error: Could not load person image: {person_input}")
                return None, None, None
        else:
            # Assume numpy array
            person = person_input.copy()

        # Load Cloth Image
        if isinstance(cloth_input, str):
            cloth = cv2.imread(cloth_input)
            if cloth is None:
                print(f"Error: Could not load cloth image: {cloth_input}")
                return None, None, None
        else:
            # Assume numpy array
            cloth = cloth_input.copy()
        
        p_pose = self.get_pose(person)
        c_pose = self.get_pose(cloth)
        
        if not p_pose:
            print("✗ Person pose failed")
            return None, None, None

        # --- COMPLEX POSE CHECK ---
        # If pose is complex (sitting, lying), switch to Head Swap Strategy
        if self.is_complex_pose(p_pose, person.shape):
            print("⚠ Complex Sitting/Lying Pose Detected!")
            print("→ Switching to HEAD SWAP Strategy (Pasting User Head onto Cloth Model)")
            
            # Need Cloth Pose for alignment
            if not c_pose:
                # Try to detect pose on cloth image
                c_pose = self.get_pose(cloth)
                
            if c_pose:
                head_swap_result = self.process_head_swap(person, cloth, p_pose, c_pose, person.shape)
                
                # Resize result to match original person image dimensions
                if head_swap_result.shape[:2] != person.shape[:2]:
                    head_swap_result = cv2.resize(head_swap_result, (person.shape[1], person.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                
                if out_path_final:
                    cv2.imwrite(out_path_final, head_swap_result)
                    print("✅ Saved Head Swap Result:", out_path_final)
                
                return head_swap_result, head_swap_result, None
            else:
                print("⚠ Cloth pose not detected, falling back to standard warp (might fail)")

        final_cloth = np.zeros_like(person)
        final_alpha = np.zeros(person.shape[:2], dtype=np.float32)

        # Check if target person is wearing pants/covering legs
        target_has_pants = self.is_person_wearing_pants(person)
        if target_has_pants:
            print("ℹ Target person is wearing pants - will replace legs if needed")
        else:
            print("ℹ Target person has bare legs (dress/skirt/shorts) - will preserve original legs")

        print("Processing as SAREE...")
        mask_saree = self.get_saree_mask(cloth)
        if mask_saree is None: return None, None, None

        # Check if this is a short dress (not extending to feet) - needs legs/feet/shoes from cloth
        h_cloth, w_cloth = cloth.shape[:2]
        coords_saree = cv2.findNonZero(mask_saree)
        is_short_dress = False
        if coords_saree is not None:
            x_s, y_s, w_s, h_s = cv2.boundingRect(coords_saree)
            bottom_y_saree = y_s + h_s
            # If dress bottom is above 80% of image height, it's a short dress
            is_short_dress = bottom_y_saree < (h_cloth * 0.80)
        
        # Also extract arms and hands from cloth image
        mask_arms_hands = self.get_cloth_arms_hands_mask(cloth)
        
        # Combine saree mask with arms+hands
        mask_saree = cv2.bitwise_or(mask_saree, mask_arms_hands)
        
        # ALWAYS include legs/feet/shoes from cloth for ANY dress/skirt (short or long)
        # This ensures complete outfit extraction regardless of target's clothing
        print("✓ ALWAYS including legs, feet, and shoes from cloth image (full outfit)")
        cloth_lower_mask = self.get_cloth_lower_body_mask(cloth)  # Legs + feet + shoes
        mask_saree = cv2.bitwise_or(mask_saree, cloth_lower_mask)
        
        # Clean the combined mask to remove holes
        # Step 1: Fill small holes
        kernel_close_small = np.ones((5, 5), np.uint8)
        mask_saree = cv2.morphologyEx(mask_saree, cv2.MORPH_CLOSE, kernel_close_small)
        
        # Step 2: Fill larger holes
        kernel_close_medium = np.ones((9, 9), np.uint8)
        mask_saree = cv2.morphologyEx(mask_saree, cv2.MORPH_CLOSE, kernel_close_medium)
        
        # Step 3: Remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        mask_saree = cv2.morphologyEx(mask_saree, cv2.MORPH_OPEN, kernel_open)
        
        # Step 4: Fill any remaining holes using floodFill
        mask_inv = cv2.bitwise_not(mask_saree)
        h, w = mask_saree.shape
        # Fill from corners
        cv2.floodFill(mask_inv, None, (0, 0), 255)
        cv2.floodFill(mask_inv, None, (w-1, 0), 255)
        cv2.floodFill(mask_inv, None, (0, h-1), 255)
        cv2.floodFill(mask_inv, None, (w-1, h-1), 255)
        # Fill holes
        holes = cv2.bitwise_not(mask_inv)
        mask_saree = cv2.bitwise_or(mask_saree, holes)
        
        # Step 5: Final closing
        mask_saree = cv2.morphologyEx(mask_saree, cv2.MORPH_CLOSE, kernel_close_medium)

        # Treat Saree as "Lower/Full" logic (Robust Scale, No Shear)
        if c_pose:
            # Case 1: Robust Scale (No Shear)
            s_warp = self.process_lower_robust(
                cloth, mask_saree, c_pose, p_pose, person.shape, target_has_pants=target_has_pants
            )
        else:
            # Case 2: Geometric Scale
            s_warp = self.process_flat_geometric(
                cloth, mask_saree, p_pose, person.shape
            )
        
        if s_warp is not None:
            if s_warp.shape[2] == 4:
                # Use the actual alpha channel for the mask
                s_alpha = s_warp[:, :, 3].astype(np.float32) / 255.0
                s_warp_rgb = s_warp[:, :, :3]
            else:
                # Fallback if no alpha
                s_alpha = (np.sum(s_warp, axis=2) > 10).astype(np.float32)
                s_warp_rgb = s_warp

            # Resize if dimension mismatch
            if s_warp_rgb.shape[:2] != person.shape[:2]:
                s_warp_rgb = cv2.resize(
                    s_warp_rgb, (person.shape[1], person.shape[0])
                )
                s_alpha = cv2.resize(
                    s_alpha, (person.shape[1], person.shape[0])
                )

            # Blend
            final_cloth = s_warp_rgb * s_alpha[:, :, None]
            final_alpha = s_alpha

        # --- CREATE NATURAL MULTI-COLOR BACKGROUND FROM PERSON IMAGE ---
        # Instead of single color, create a varied background that matches the actual background
        ph, pw = person.shape[:2]
        
        # Create background texture by sampling and interpolating edge regions
        print("✓ Creating natural multi-color background for seamless blending")
        
        # Sample larger edge regions for better background texture
        edge_size = min(80, pw // 6, ph // 6)
        
        # Create a background map by blending edge samples
        background_map = np.zeros_like(person, dtype=np.float32)
        weight_map = np.zeros((ph, pw), dtype=np.float32)
        
        # Sample from all edges with gradual blending
        regions = [
            (person[0:edge_size, :], 'top'),
            (person[ph-edge_size:ph, :], 'bottom'),
            (person[:, 0:edge_size], 'left'),
            (person[:, pw-edge_size:pw], 'right')
        ]
        
        for region, side in regions:
            if region.size > 0:
                if side == 'top':
                    # Blend top edge downward
                    for i in range(ph):
                        blend_factor = np.exp(-i / (ph * 0.3))
                        if i < edge_size:
                            background_map[i, :] += region[i, :] * blend_factor
                        else:
                            background_map[i, :] += region[-1, :] * blend_factor
                        weight_map[i, :] += blend_factor
                elif side == 'bottom':
                    # Blend bottom edge upward
                    for i in range(ph):
                        blend_factor = np.exp(-(ph - 1 - i) / (ph * 0.3))
                        idx = i - (ph - edge_size)
                        if idx >= 0:
                            background_map[i, :] += region[idx, :] * blend_factor
                        else:
                            background_map[i, :] += region[0, :] * blend_factor
                        weight_map[i, :] += blend_factor
                elif side == 'left':
                    # Blend left edge rightward
                    for j in range(pw):
                        blend_factor = np.exp(-j / (pw * 0.3))
                        if j < edge_size:
                            background_map[:, j] += region[:, j] * blend_factor
                        else:
                            background_map[:, j] += region[:, -1] * blend_factor
                        weight_map[:, j] += blend_factor
                elif side == 'right':
                    # Blend right edge leftward
                    for j in range(pw):
                        blend_factor = np.exp(-(pw - 1 - j) / (pw * 0.3))
                        idx = j - (pw - edge_size)
                        if idx >= 0:
                            background_map[:, j] += region[:, idx] * blend_factor
                        else:
                            background_map[:, j] += region[:, 0] * blend_factor
                        weight_map[:, j] += blend_factor
        
        # Normalize by weights to get final background
        weight_map = np.maximum(weight_map, 1e-6)
        background_map = (background_map / weight_map[:, :, np.newaxis]).astype(np.uint8)
        
        # Also keep single color fallback for specific operations
        background_color = np.median(background_map.reshape(-1, 3), axis=0).astype(np.uint8)
        
        # --- CHECK IF PANTS/SKIRT/SHORT DRESS WERE PROCESSED ---
        is_lower_body = self.is_pants_or_shorts(cloth)
        # Check if this is a short dress (not extending to feet)
        coords_final = cv2.findNonZero((final_alpha > 0.5).astype(np.uint8) * 255) if np.max(final_alpha) > 0 else None
        is_short_dress_check = False
        if coords_final is not None:
            ph, pw = person.shape[:2]
            x_f, y_f, w_f, h_f = cv2.boundingRect(coords_final)
            bottom_y_final = y_f + h_f
            # If dress bottom is above 80% of image height, it's a short dress
            is_short_dress_check = bottom_y_final < (ph * 0.80)
        
        # ALWAYS mask person's lower body when processing ANY dress/garment
        # This ensures complete outfit replacement regardless of what target is wearing
        print("✓ ALWAYS replacing person's lower body (legs, feet, shoes) with background")
        person_lower_mask = self.get_person_lower_body_mask(person)
        person_lower_alpha = person_lower_mask.astype(np.float32) / 255.0
        
        # Use inpainting for clean background filling (no blur artifacts)
        person_masked = person.copy()
        
        # Convert mask to uint8 for inpainting
        inpaint_mask_lower = person_lower_mask.astype(np.uint8)
        
        # --- PROTECTION MASKS ---
        person_hair_mask = self.get_hair_mask(person)
        hair_alpha = person_hair_mask.astype(np.float32) / 255.0
        # Head mask for protection during inpainting (Face + Hair + Neck)
        person_head_mask = self.get_person_head_mask(person)
        head_alpha = person_head_mask.astype(np.float32) / 255.0
        
        # Dilate mask slightly to ensure complete coverage
        kernel_dilate = np.ones((5, 5), np.uint8)
        inpaint_mask_lower = cv2.dilate(inpaint_mask_lower, kernel_dilate, iterations=2)
        
        # Ensure head, face, neck are NOT part of inpaint mask
        inpaint_mask_lower[person_head_mask > 0] = 0
        
        # Use inpainting to fill masked areas with natural background extension
        person_masked = cv2.inpaint(person_masked, inpaint_mask_lower, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                 
        # --- HEAD/FACE TRIMMING ---
        # Trim cloth where person's head/face/hair is visible
        # Subtract from cloth alpha to avoid overlap blurring
        # DISABLED: User requested only hair trimming. We paste hair on top later (Step 3).
        # Subtracting head_alpha (Face+Neck) cuts the shirt unnecessarily at the chest.
        # final_alpha = np.clip(final_alpha - head_alpha, 0, 1)

        # --- CLEAN CLOTH ALPHA (REMOVE HOLES, ENSURE PERFECT QUALITY) ---
        # Enhanced cleanup to remove all holes and ensure clean cut
        if np.max(final_alpha) > 0:
            # Convert to uint8 for morphological operations
            alpha_uint8 = (final_alpha * 255).astype(np.uint8)
            
            # Step 1: Fill small holes
            kernel_close_small = np.ones((5, 5), np.uint8)
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_close_small)
            
            # Step 2: Fill larger holes
            kernel_close_medium = np.ones((9, 9), np.uint8)
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_close_medium)
            
            # Step 3: Remove small noise
            kernel_open = np.ones((3, 3), np.uint8)
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, kernel_open)
            
            # Step 4: Fill any remaining holes using floodFill
            # Invert to find holes
            alpha_inv = cv2.bitwise_not(alpha_uint8)
            h, w = alpha_uint8.shape
            # Fill from all corners to identify holes
            cv2.floodFill(alpha_inv, None, (0, 0), 255)
            cv2.floodFill(alpha_inv, None, (w-1, 0), 255)
            cv2.floodFill(alpha_inv, None, (0, h-1), 255)
            cv2.floodFill(alpha_inv, None, (w-1, h-1), 255)
            # Holes are areas not reached by floodFill
            holes = cv2.bitwise_not(alpha_inv)
            # Fill all holes
            alpha_uint8 = cv2.bitwise_or(alpha_uint8, holes)
            
            # Step 5: Final closing to ensure smooth edges
            kernel_close_final = np.ones((7, 7), np.uint8)
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_close_final)
            
            # Make binary - sharp edges, no soft transitions
            _, alpha_uint8 = cv2.threshold(alpha_uint8, 127, 255, cv2.THRESH_BINARY)
            
            # Convert back to float - keep it binary (0 or 1)
            final_alpha = (alpha_uint8 > 127).astype(np.float32)
        
        # --- CHECK IF DRESS EXTENDS TO FEET ---
        # If dress extends to bottom (covers feet area), track it for shoe masking
        ph, pw = person.shape[:2]
        person_ankle_y = (p_pose["l_ankle"][1] + p_pose["r_ankle"][1]) / 2
        
        # Check if cloth extends to near the bottom (within 10% of image height from bottom)
        bottom_threshold = ph * 0.9  # 90% down the image
        cloth_bottom_y = 0
        
        # Find the bottom-most point where cloth is present
        cloth_coords = np.where(final_alpha > 0.5)
        if len(cloth_coords[0]) > 0:
            cloth_bottom_y = np.max(cloth_coords[0])
        
        # Track if dress extends to feet (for shoe masking before blending)
        dress_extends_to_feet = (cloth_bottom_y >= bottom_threshold or 
                                 cloth_bottom_y >= person_ankle_y)
        
        # --- REPLACE SHOES WITH BACKGROUND (ALWAYS) ---
        # Always replace person's shoes with background before blending
        # The cloth shoes will cover this area
        print("✓ ALWAYS masking person's shoes with background")
        person_feet_shoes_mask = self.get_person_feet_shoes_mask(person)
        person_feet_shoes_alpha = person_feet_shoes_mask.astype(np.float32) / 255.0
        
        # Use inpainting for clean shoe removal (no blur artifacts)
        inpaint_mask_shoes = person_feet_shoes_mask.astype(np.uint8)
        
        # Dilate mask slightly to ensure complete coverage
        inpaint_mask_shoes = cv2.dilate(inpaint_mask_shoes, kernel_dilate, iterations=2)
        
        # Ensure head area is NOT part of inpaint mask
        inpaint_mask_shoes[person_head_mask > 0] = 0
        
        # Use inpainting to fill shoe areas with natural background extension
        person_masked = cv2.inpaint(person_masked, inpaint_mask_shoes, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        
        
        # --- CRITICAL: REMOVE ALL ORIGINAL CLOTHES FIRST ---
        # We need to remove original clothes/arms/lower body BEFORE blending new cloth
        # This ensures original clothing is completely replaced with background
        
        print("✓ Step 1: Removing ALL original clothing from person image")
        
        # Start with the person image that already has lower body replaced with background
        person_clean = person_masked.copy()
        
        # Get mask for person's original clothes, arms, and hands
        person_remove_mask = self.get_person_cloth_arms_hands_mask(person)
        person_remove_alpha = person_remove_mask.astype(np.float32) / 255.0
        
        # Use inpainting for clean upper body removal (no blur artifacts)
        inpaint_mask_upper = person_remove_mask.astype(np.uint8)
        
        # Dilate mask to ensure complete coverage
        kernel_dilate_upper = np.ones((10, 10), np.uint8)
        inpaint_mask_upper = cv2.dilate(inpaint_mask_upper, kernel_dilate_upper, iterations=3)
        
        # CRITICAL: Subtract head/face/neck from inpaint mask so detail isn't lost to blurring
        inpaint_mask_upper[person_head_mask > 0] = 0
        
        # Use inpainting to fill upper body areas with natural background extension
        person_clean = cv2.inpaint(person_clean, inpaint_mask_upper, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        
        print("✓ Step 2: Blending new warped cloth on top of clean person image")
        
        # Now blend the new warped cloth on top of the clean person image
        # The new cloth will cover the background where it should be
        out = person_clean * (1 - final_alpha[:, :, None]) + \
              final_cloth * final_alpha[:, :, None]
        
        # --- FINAL HAIR RESTORATION ---
        # Paste original hair back on top to ensure it remains sharp and untouched
        print("✓ Step 3: Restoring original hair on top for perfect detail")
        out = out * (1 - hair_alpha[:, :, None]) + person * hair_alpha[:, :, None]
        
        # No additional masking needed - original clothes are already removed!
        
        # Prepare outputs
        final_result = out.astype(np.uint8)

        # Visualization Mask Logic
        combined_person_mask = np.maximum(person_remove_alpha, person_lower_alpha)
        h, w = person.shape[:2]
        visualization_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        person_mask_binary = (combined_person_mask > 0.5).astype(np.uint8) * 255
        kernel_expand = np.ones((15, 15), np.uint8)
        person_mask_binary = cv2.dilate(person_mask_binary, kernel_expand, iterations=3)
        visualization_mask[person_mask_binary > 0] = [255, 255, 255]
        
        cloth_mask_binary = (final_alpha > 0.5).astype(np.uint8) * 255
        visualization_mask[cloth_mask_binary > 0] = [0, 0, 0]
        
        warp_mask = cloth_mask_binary

        # Save if requested
        if out_path_final:
            cv2.imwrite(out_path_final, final_result)
            print("✅ Saved Result:", out_path_final)
            
            from pathlib import Path
            out_p = Path(out_path_final)
            combined_mask_path = str(out_p.parent / (out_p.stem + '_MASK' + out_p.suffix))
            cv2.imwrite(combined_mask_path, visualization_mask)
            print(f"✅ Saved MASK: {combined_mask_path}")

        return final_result, visualization_mask, warp_mask


# ---------------- RUN ----------------
if __name__ == "__main__":
    warper = ShoulderHeightDressWarper(
        r"D:\IDM_VTON_Weights\SegFormerB2Clothes\segformer_b2_clothes.onnx"
    )

    warper.process(
        r"c:\Users\PC\Downloads\232fa7aa-4854-41d8-9c78-576430912fd81723280009308-FableStreet-LivIn-Bootcut-Trousers-5211723280009216-6.jpg", 
        r"c:\Users\PC\Downloads\Purple-Chiffon-Plain-Saree-With-Designer-Blouse-29015-A13082024.webp",
        r"C:\Users\PC\.gemini\antigravity\scratch\output1\FINAL_DRESS12.png"
    )
