import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import sys
import os
from pathlib import Path

# Add project root to sys.path and change working directory to project root
root_path = Path(__file__).resolve().parent.parent.parent
os.chdir(str(root_path))
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import onnxruntime as ort
import mediapipe as mp
from extras.inpaint_mask import generate_mask_from_image, SAMOptions


class ClothMasker:
    """
    Creates thick masking around cloth in the result image from robust_cloth_warp.py
    Uses SegFormer b3-fashion model to segment cloth areas.
    """
    
    def __init__(self, model_path, onnx_model_path, b2_onnx_path=None):
        """
        Initialize with SegFormer b3-fashion model and LIP parsing model for arms/hands.
        
        Args:
            model_path: Path to segformer-b3-fashion model directory
            onnx_model_path: Path to LIP (Look Into Person) parsing ONNX model
            b2_onnx_path: Path to B2 clothes parsing ONNX model
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SegFormer b3-fashion model from {model_path} on {self.device}...")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_path)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
            self.model.eval()
            print("[OK] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
        
        # Load LIP parsing ONNX model for arms/hands detection
        print(f"Loading LIP parsing ONNX model from {onnx_model_path}...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.onnx_session = ort.InferenceSession(str(onnx_model_path), sess_options=sess_opts, providers=providers)
            print(f"LIP ONNX Providers: {self.onnx_session.get_providers()}")
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            print("[OK] LIP parsing ONNX model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model: {e}")
            raise
        
        # Initialize MediaPipe Pose for backup detection
        print("Loading MediaPipe Pose for backup detection...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)
        print("[OK] MediaPipe Pose loaded successfully")

        # Load B2 clothes parsing ONNX model if provided
        self.b2_session = None
        if b2_onnx_path:
            print(f"Loading B2 clothes parsing ONNX model from {b2_onnx_path}...")
            try:
                self.b2_session = ort.InferenceSession(str(b2_onnx_path), sess_options=sess_opts, providers=providers)
                self.b2_input_name = self.b2_session.get_inputs()[0].name
                print("[OK] B2 clothes parsing ONNX model loaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to load B2 ONNX model: {e}")
                # Don't raise, just disable v3 if b2 is missing
    
    
    def get_cloth_mask(self, image):
        """
        Extract cloth mask using SegFormer b3-fashion.
        
        Returns binary mask where cloth pixels are 255, background is 0.
        """
        # Prepare image for model
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu()
        
        # Resize logits to original image size
        logits = torch.nn.functional.interpolate(
            logits, 
            size=image.shape[:2], 
            mode="bilinear", 
            align_corners=False
        )
        
        # Get segmentation map
        seg = logits.argmax(dim=1)[0].numpy()
        
        # Store segmentation map for later use
        self.seg_map = seg
        
        # Saree/Cloth Union Classes (All garment-related classes)
        # Exclude: 0 (Unlabelled), 14 (glasses), 15 (hat), 16 (headband/hair), 
        # 24 (shoe), 25 (bag/wallet), 27 (umbrella)
        # Include: All garments, parts, decorations, and fabric
        cloth_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
            17, 18, 19, 20, 21, 22, 23, 
            26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
        ]
        
        # Create binary mask
        mask = np.isin(seg, cloth_ids).astype(np.uint8) * 255
        
        return mask
    
    def get_b2_cloth_mask(self, image):
        """
        Extract cloth mask using B2 model.
        Returns binary mask where cloth pixels are 255.
        """
        if self.b2_session is None:
            return None
            
        h, w = image.shape[:2]
        img = cv2.resize(image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)[None].astype(np.float32)
        
        logits = self.b2_session.run(None, {self.b2_input_name: img})[0][0]
        logits = cv2.resize(logits.transpose(1, 2, 0), (w, h))
        seg = np.argmax(logits, axis=2)
        
        # B2 Cloth Classes: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 18: ?? (Let's stick to 4,5,6,7,8)
        cloth_ids = [4, 5, 6, 7, 8]
        mask = np.isin(seg, cloth_ids).astype(np.uint8) * 255
        
        return mask

    def get_exposed_skin_mask(self, image):
        """
        Extract exposed skin (arms only) using LIP parsing model.
        LIP Classes: 14 (Left-arm), 15 (Right-arm)
        
        Returns binary mask where exposed arms are 255.
        """
        h, w = image.shape[:2]
        
        # Prepare image for ONNX LIP model
        img = cv2.resize(image, (473, 473))  # LIP model uses 473x473 input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)[None].astype(np.float32)
        
        # Run inference
        logits = self.onnx_session.run(None, {self.onnx_input_name: img})[0][0]
        logits = cv2.resize(logits.transpose(1, 2, 0), (w, h))
        
        # Get segmentation map
        seg = np.argmax(logits, axis=2)
        
        # Debug: Print unique classes detected
        unique_classes = np.unique(seg)
        print(f"[DEBUG] LIP model detected classes: {unique_classes}")
        
        # LIP model class mapping (CORRECT):
        # 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes
        # 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe
        # 11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm
        # 16: Bag, 17: Scarf
        
        # Include only arms (no legs)
        arms_hands_classes = {14, 15}  # Left-arm, Right-arm
        
        # Create mask for arms and hands
        skin_mask = np.isin(seg, list(arms_hands_classes)).astype(np.uint8) * 255
        
        # Debug: Count pixels for each arm class
        for cls in arms_hands_classes:
            count = np.sum(seg == cls)
            print(f"[DEBUG] Class {cls} pixels: {count}")
        
        print(f"[DEBUG] Initial arm/hand mask pixels: {cv2.countNonZero(skin_mask)}")
        
        # Save segmentation visualization for debugging
        seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in unique_classes:
            if cls in [14, 15]:  # Arms
                seg_vis[seg == cls] = [0, 255, 0]  # Green for arms
            elif cls in [12, 13]:  # Legs (CORRECT indices)
                seg_vis[seg == cls] = [0, 255, 255]  # Cyan for legs
            elif cls == 6:  # Pants
                seg_vis[seg == cls] = [255, 0, 255]  # Magenta for pants
            elif cls in [9, 10]:  # Shoes
                seg_vis[seg == cls] = [255, 255, 0]  # Yellow for shoes
            elif cls == 11:  # Face
                seg_vis[seg == cls] = [255, 0, 0]  # Blue for face
            elif cls in [4, 5, 7]:  # Clothes
                seg_vis[seg == cls] = [0, 0, 255]  # Red for clothes
        
        self.seg_visualization = seg_vis
        self.lip_seg = seg  # Cache LIP segmentation for v2 mask
        
        # Create separate visualizations for hands and legs
        # Hands visualization (if detected)
        hands_vis = np.zeros((h, w, 3), dtype=np.uint8)
        # Note: LIP model doesn't have separate hand class, hands are part of arms
        # We'll mark the lower part of arms as hands for visualization
        self.hands_visualization = hands_vis
        
        # Legs visualization
        legs_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in [12, 13]:  # Left-leg, Right-leg
            if cls in unique_classes:
                legs_vis[seg == cls] = [0, 255, 255]  # Cyan for legs
        self.legs_visualization = legs_vis
        
        # Aggressive cleaning and expansion to ensure full arm and hand coverage
        # First close small holes
        kernel_close_small = np.ones((9, 9), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close_small)
        
        # Close larger holes
        kernel_close_large = np.ones((15, 15), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close_large)
        
        # Remove tiny noise
        kernel_open = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Dilate significantly to ensure full arm and hand coverage (including fingers)
        kernel_dilate = np.ones((20, 20), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel_dilate, iterations=3)
        
        # Fill any remaining holes aggressively - REMOVED
        # This was causing issues where arm loops were filled in
        # mask_inv = cv2.bitwise_not(skin_mask)
        # cv2.floodFill(mask_inv, None, (0, 0), 255)
        # cv2.floodFill(mask_inv, None, (w-1, 0), 255)
        # cv2.floodFill(mask_inv, None, (0, h-1), 255)
        # cv2.floodFill(mask_inv, None, (w-1, h-1), 255)
        # holes = cv2.bitwise_not(mask_inv)
        # skin_mask = cv2.bitwise_or(skin_mask, holes)
        
        # Final closing to ensure no small holes remain
        kernel_final = np.ones((11, 11), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_final)
        
        print(f"[DEBUG] Final arm/hand mask pixels (after morphology): {cv2.countNonZero(skin_mask)}")
        
        return skin_mask
    
    def get_face_hair_protection_mask(self, image):
        """
        Extract face and hair mask to protect these areas.
        Uses LIP parsing model + MediaPipe Pose (fallback for face)
        
        LIP Classes: 2 (Hair), 11 (Face)
        MediaPipe: Landmarks 0-10 (Face)
        
        Returns binary mask where face and hair pixels are 255 (to be excluded from masking).
        """
        h, w = image.shape[:2]
        
        # --- Method 1: LIP Model ---
        # Prepare image for ONNX LIP model
        img = cv2.resize(image, (473, 473))  # LIP model uses 473x473 input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)[None].astype(np.float32)
        
        # Run inference
        logits = self.onnx_session.run(None, {self.onnx_input_name: img})[0][0]
        logits = cv2.resize(logits.transpose(1, 2, 0), (w, h))
        
        # Get segmentation map
        seg = np.argmax(logits, axis=2)
        
        # LIP class mapping: 2: Hair, 11: Face
        face_class = 11
        hair_class = 2
        
        # Create masks from LIP
        lip_face_mask = (seg == face_class).astype(np.uint8) * 255
        lip_hair_mask = (seg == hair_class).astype(np.uint8) * 255
        
        print(f"[DEBUG] LIP Face pixels: {cv2.countNonZero(lip_face_mask)}")
        print(f"[DEBUG] LIP Hair pixels: {cv2.countNonZero(lip_hair_mask)}")

        # --- Method 2: MediaPipe Face Fallback ---
        # Use existing MediaPipe pose landmarks if available, or run new inference
        mp_face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process with MediaPipe if we haven't already for this image context, 
        # but self.pose.process is cheap enough or we can try to reuse.
        # Ideally we'd reuse, but for safety let's run it on the current image.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Face landmarks indices in Pose model: 0-10
            # 0: nose, 1-3: right eye, 4-6: left eye, 7: right ear, 8: left ear, 9: mouth_left, 10: mouth_right
            face_pts = []
            for idx in range(11):
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if lm.visibility > 0.5:
                        face_pts.append((int(lm.x * w), int(lm.y * h)))
            
            if len(face_pts) > 3:
                # distinct points found
                # Draw a convex hull filled mask for the face
                hull = cv2.convexHull(np.array(face_pts))
                cv2.fillConvexPoly(mp_face_mask, hull, 255)
                # Dilate the MediaPipe face mask generously to cover cheeks/forehead not covered by landmarks
                # Landmarks are mostly central features (eyes, nose, mouth)
                mp_kernel = np.ones((40, 40), np.uint8) 
                mp_face_mask = cv2.dilate(mp_face_mask, mp_kernel, iterations=1)
                print(f"[DEBUG] MediaPipe Face pixels: {cv2.countNonZero(mp_face_mask)}")
        
        # Combine Face Masks
        face_mask = cv2.bitwise_or(lip_face_mask, mp_face_mask)
        
        # Dilate combined face mask for safety
        kernel_face = np.ones((15, 15), np.uint8)
        face_mask = cv2.dilate(face_mask, kernel_face, iterations=1)
        
        # --- Hair Processing ---
        # Reduce erosion to protect more hair. 
        # Previous was (15,15) iter=2 -> too aggressive.
        # New: (5,5) iter=1 just to trim slightly fuzzy edges, or skip erosion if user wants FULL protection.
        # User said "protect... also on eth ahirs", implying high protection.
        # We will do a very light erosion to avoid jagged edges but keep most hair.
        kernel_hair_erode = np.ones((5, 5), np.uint8)
        hair_mask = cv2.erode(lip_hair_mask, kernel_hair_erode, iterations=1)
        
        # Keep only the largest hair component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hair_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            hair_mask = (labels == largest_label).astype(np.uint8) * 255
            
        print(f"[DEBUG] Final Face pixels: {cv2.countNonZero(face_mask)}")
        print(f"[DEBUG] Final Hair pixels: {cv2.countNonZero(hair_mask)}")
        
        # Combine
        protection_mask = cv2.bitwise_or(face_mask, hair_mask)
        
        # Clean up
        kernel_clean = np.ones((7, 7), np.uint8)
        protection_mask = cv2.morphologyEx(protection_mask, cv2.MORPH_CLOSE, kernel_clean)
        
        print(f"[OK] Face/Hair protection mask: {cv2.countNonZero(protection_mask)} pixels")
        
        return protection_mask
    
    def get_mediapipe_limb_mask(self, image, cloth_mask=None):
        """
        Use MediaPipe Pose to detect arms, hands, legs, and feet as backup.
        This supplements the LIP model to catch anything it missed.
        
        Args:
            image: Input image
            cloth_mask: Optional cloth mask to exclude areas covered by cloth
        """
        h, w = image.shape[:2]
        
        # Calculate sizes proportional to image dimensions (resolution-independent)
        base_size = min(h, w)
        
        # Arm sizing (INCREASED for better hand coverage)
        arm_circle_radius = int(base_size * 0.035)  # Increased from 2.5% to 3.5%
        arm_line_thickness = int(base_size * 0.045)  # Increased from 3.0% to 4.5%
        arm_dilation_kernel = max(7, int(base_size * 0.018))  # Increased from 1.2% to 1.8%
        
        # Hand-specific sizing (larger for finger coverage)
        hand_circle_radius = int(base_size * 0.045)  # 4.5% for hand joints
        
        # Leg sizing (BALANCED - full coverage but not too wide)
        leg_circle_radius = int(base_size * 0.055)  # 5.5% (balanced)
        leg_line_thickness = int(base_size * 0.070)  # 7.0% (balanced)
        leg_dilation_kernel = max(10, int(base_size * 0.020))  # 2.0% (moderate)
        leg_width = int(base_size * 0.080)  # 8.0% for gap filling (balanced)
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_image)
        
        # Create separate masks for arms and legs
        arm_mask = np.zeros((h, w), dtype=np.uint8)
        leg_mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Define body part connections for masking
            # Arms: shoulders -> elbows -> wrists -> fingers
            left_arm = [11, 13, 15, 17, 19, 21]  # left shoulder -> wrist -> hand
            right_arm = [12, 14, 16, 18, 20, 22]  # right shoulder -> wrist -> hand
            
            # Legs: Include hips to fill gap between cloth and knees
            left_leg = [23, 25, 27, 29, 31]  # left hip -> knee -> ankle -> foot
            right_leg = [24, 26, 28, 30, 32]  # right hip -> knee -> ankle -> foot
            
            all_limbs = left_arm + right_arm + left_leg + right_leg
            
            # Convert landmarks to pixel coordinates and draw circles
            for idx in all_limbs:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if lm.visibility > 0.5:  # Only use visible landmarks
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        
                        # Different sizes for arms vs legs (resolution-independent)
                        if idx in left_leg + right_leg:
                            # Large radius for legs
                            cv2.circle(leg_mask, (x, y), leg_circle_radius, 255, -1)
                        elif idx in [17, 19, 21, 18, 20, 22]:  # Hand landmarks (pinky, index, thumb)
                            # Extra large radius for hands to cover fingers
                            cv2.circle(arm_mask, (x, y), hand_circle_radius, 255, -1)
                        else:
                            # Standard radius for arms
                            cv2.circle(arm_mask, (x, y), arm_circle_radius, 255, -1)
            
            # Draw connections between landmarks with different masks
            arm_connections = [
                (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22)   # Right arm
            ]
            
            leg_connections = [
                (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
                (24, 26), (26, 28), (28, 30), (28, 32)   # Right leg
            ]
            
            # Draw arm connections with thin lines (resolution-independent)
            for start_idx, end_idx in arm_connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_lm = landmarks[start_idx]
                    end_lm = landmarks[end_idx]
                    if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                        start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                        end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                        cv2.line(arm_mask, start_pt, end_pt, 255, arm_line_thickness)
            
            # Draw leg connections with thick lines (resolution-independent)
            for start_idx, end_idx in leg_connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_lm = landmarks[start_idx]
                    end_lm = landmarks[end_idx]
                    if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                        start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                        end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                        cv2.line(leg_mask, start_pt, end_pt, 255, leg_line_thickness)
            
            # Apply different dilation for arms and legs (resolution-independent)
            # Larger dilation for arms to cover hands and fingers
            arm_kernel = np.ones((arm_dilation_kernel, arm_dilation_kernel), np.uint8)
            arm_mask = cv2.dilate(arm_mask, arm_kernel, iterations=3)  # Increased from 1 to 3
            
            # Additional closing to fill holes in hands/arms
            kernel_arm_close = np.ones((int(base_size * 0.025), int(base_size * 0.025)), np.uint8)
            arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, kernel_arm_close)
            
            # Moderate dilation for legs to fill gaps
            leg_kernel = np.ones((leg_dilation_kernel, leg_dilation_kernel), np.uint8)
            leg_mask = cv2.dilate(leg_mask, leg_kernel, iterations=2)  # Increased back to 2
            
            # If cloth mask is provided, find cloth bottom edge and extend leg mask
            if cloth_mask is not None:
                # Find the bottom edge of the cloth for each leg region
                # Get hip and knee positions to determine leg regions
                if len(landmarks) > 26:
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                    left_knee = landmarks[25]
                    right_knee = landmarks[26]
                    
                    if left_knee.visibility > 0.5 and left_hip.visibility > 0.5:
                        # Left leg region
                        lk_x, lk_y = int(left_knee.x * w), int(left_knee.y * h)
                        lh_x, lh_y = int(left_hip.x * w), int(left_hip.y * h)
                        
                        # Search upward from knee to find cloth bottom edge
                        for y in range(lk_y, max(0, lh_y - 50), -1):
                            x_range = range(max(0, lk_x - leg_width), min(w, lk_x + leg_width))
                            if any(cloth_mask[y, x] > 0 for x in x_range):
                                # Found cloth edge, fill down to below knee
                                fill_bottom = lk_y + int(base_size * 0.06)
                                cv2.rectangle(leg_mask, (lk_x - leg_width, y), (lk_x + leg_width, fill_bottom), 255, -1)
                                # Ellipse for natural leg shape
                                center = (lk_x, (y + fill_bottom) // 2)
                                axes = (int(leg_width * 0.8), (fill_bottom - y) // 2)
                                cv2.ellipse(leg_mask, center, axes, 0, 0, 360, 255, -1)
                                break
                    
                    if right_knee.visibility > 0.5 and right_hip.visibility > 0.5:
                        # Right leg region
                        rk_x, rk_y = int(right_knee.x * w), int(right_knee.y * h)
                        rh_x, rh_y = int(right_hip.x * w), int(right_hip.y * h)
                        
                        # Search upward from knee to find cloth bottom edge
                        for y in range(rk_y, max(0, rh_y - 50), -1):
                            x_range = range(max(0, rk_x - leg_width), min(w, rk_x + leg_width))
                            if any(cloth_mask[y, x] > 0 for x in x_range):
                                # Found cloth edge, fill down to below knee
                                fill_bottom = rk_y + int(base_size * 0.06)
                                cv2.rectangle(leg_mask, (rk_x - leg_width, y), (rk_x + leg_width, fill_bottom), 255, -1)
                                # Ellipse for natural leg shape
                                center = (rk_x, (y + fill_bottom) // 2)
                                axes = (int(leg_width * 0.8), (fill_bottom - y) // 2)
                                cv2.ellipse(leg_mask, center, axes, 0, 0, 360, 255, -1)
                                break
            
            # Combine arm and leg masks
            mp_mask = cv2.bitwise_or(arm_mask, leg_mask)
            
            # Store separate visualizations for arms/hands and legs
            self.mp_arms_hands_visualization = arm_mask
            self.mp_legs_visualization = leg_mask
            
            # After filling gaps, remove cloth-covered areas
            if cloth_mask is not None:
                # Only keep parts NOT covered by cloth
                cloth_inverse = cv2.bitwise_not(cloth_mask)
                mp_mask = cv2.bitwise_and(mp_mask, cloth_inverse)
        
        return mp_mask
    
    def _connect_masks_from_below(self, warp_mask, legs_mask, h, w):
        """
        Fill the gap BETWEEN the legs horizontally at the bottom.
        Does NOT extend to image bottom, only fills the gap between legs.
        
        Args:
            warp_mask: Mask from robust_cloth_warp.py
            legs_mask: Legs mask from detection
            h, w: Image dimensions
            
        Returns:
            Connection mask for the gap between legs
        """
        connection_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine both masks to find leg positions
        combined_temp = cv2.bitwise_or(warp_mask, legs_mask)
        
        # Find the bottom region (lower 40% of image where legs/feet are)
        lower_region_start = int(h * 0.6)
        
        # For each row in the bottom region, fill horizontally ONLY between leg positions
        for y in range(lower_region_start, h):
            row = combined_temp[y, :]
            points = np.where(row > 0)[0]
            
            if len(points) >= 2:
                # Fill horizontally between leftmost and rightmost points (between legs)
                left_x = np.min(points)
                right_x = np.max(points)
                connection_mask[y, left_x:right_x+1] = 255
        
        # Remove any extensions beyond the legs - keep only the gap
        # Find the actual bottom of legs
        leg_coords = np.where(combined_temp > 0)
        if len(leg_coords[0]) > 0:
            actual_bottom = np.max(leg_coords[0])
            # Remove everything below actual legs + small margin
            margin = int(h * 0.02)  # 2% margin
            connection_mask[actual_bottom + margin:, :] = 0
        
        # Smooth the connection
        kernel = np.ones((7, 15), np.uint8)  # Horizontal smoothing
        connection_mask = cv2.morphologyEx(connection_mask, cv2.MORPH_CLOSE, kernel)
        
        print(f"   - Connected gap between legs: {cv2.countNonZero(connection_mask)} pixels")
        
        return connection_mask
    
    def create_thick_border_mask(self, mask, border_thickness=20):
        """
        Create a thick border around the cloth mask.
        Border extends both outward and inward from cloth edge.
        
        Args:
            mask: Binary cloth mask (255 = cloth, 0 = background)
            border_thickness: Thickness of border in pixels
            
        Returns:
            Border mask (255 where border should be drawn)
        """
        # Dilate mask to create outer boundary - extends outward
        kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness*2, border_thickness*2))
        dilated = cv2.dilate(mask, kernel_outer, iterations=1)
        
        # Erode mask to create inner boundary - extends more inward into cloth
        kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(border_thickness*1.5), int(border_thickness*1.5)))
        eroded = cv2.erode(mask, kernel_inner, iterations=1)
        
        # Border extends both outward and inward from cloth edge
        border_mask = cv2.subtract(dilated, eroded)
        
        return border_mask
    
    def apply_overlay(self, image, mask, color=(255, 140, 100), alpha=0.5):
        """
        Apply colored overlay to masked regions.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask (255 = overlay area, 0 = no overlay)
            color: BGR color for overlay (default: coral/pink)
            alpha: Transparency (0.0 = transparent, 1.0 = opaque)
            
        Returns:
            Image with overlay applied
        """
        # Create colored overlay
        overlay = image.copy()
        overlay[mask > 0] = color
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def process(self, input_image_path, output_path, border_thickness=20, 
                overlay_color=(100, 140, 255), alpha=0.5, warp_mask_path=None, generate_v3=True):
        """
        Main processing pipeline: Load image -> Segment cloth -> Create thick border -> Apply overlay
        
        Args:
            input_image_path: Path to result image from robust_cloth_warp.py
            output_path: Path to save masked output
            border_thickness: Thickness of border outline in pixels (default: 20)
            overlay_color: BGR color for overlay (default: coral/pink)
            alpha: Overlay transparency (default: 0.5)
            warp_mask_path: Optional path to _MASK.png from robust_cloth_warp.py
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_image_path}")
        print(f"{'='*60}")
        
        # Load image
        image = cv2.imread(str(input_image_path))
        if image is None:
            print(f"[ERROR] Failed to load image: {input_image_path}")
            return
        
        h, w = image.shape[:2]
        print(f"[OK] Image loaded: {w}x{h}")
        
        # Get cloth mask
        print(">> Segmenting cloth regions...")
        cloth_mask = self.get_cloth_mask(image)
        
        # Clean mask (remove holes, smooth edges)
        print(">> Cleaning mask (light closing to preserve gaps)...")
        kernel_close = np.ones((3, 3), np.uint8)  # Reduced from 9x9 -> 3x3
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = np.ones((3, 3), np.uint8)   # Reduced from 5x5 -> 3x3
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Fill holes using floodFill - REMOVED for safety
        # mask_inv = cv2.bitwise_not(cloth_mask)
        # cv2.floodFill(mask_inv, None, (0, 0), 255)
        # cv2.floodFill(mask_inv, None, (w-1, 0), 255)
        # cv2.floodFill(mask_inv, None, (0, h-1), 255)
        # cv2.floodFill(mask_inv, None, (w-1, h-1), 255)
        # holes = cv2.bitwise_not(mask_inv)
        # cloth_mask = cv2.bitwise_or(cloth_mask, holes)
        
        print(f"[OK] Cloth pixels detected: {cv2.countNonZero(cloth_mask)}")
        
        # Get exposed skin (arms only) from LIP model
        print(">> Detecting exposed skin (arms only) with LIP model...")
        skin_mask = self.get_exposed_skin_mask(image)
        print(f"[OK] LIP model detected: {cv2.countNonZero(skin_mask)} pixels")
        
        # Get MediaPipe backup detection for limbs (excluding cloth-covered areas)
        print(">> Using MediaPipe Pose for backup detection (hands, arms, legs, feet)...")
        mp_mask = self.get_mediapipe_limb_mask(image, cloth_mask)
        self.mp_limb_mask = mp_mask  # Store for later saving
        print(f"[OK] MediaPipe detected: {cv2.countNonZero(mp_mask)} pixels")
        
        # Combine both masks (union) to catch everything
        combined_skin_mask = cv2.bitwise_or(skin_mask, mp_mask)
        print(f"[OK] Combined (LIP + MediaPipe): {cv2.countNonZero(combined_skin_mask)} pixels")
        
        # Update skin_mask to be the combined version
        skin_mask = combined_skin_mask
        
        # Get face and hair protection mask
        print(">> Detecting face and hair areas to protect...")
        face_hair_protection = self.get_face_hair_protection_mask(image)
        
        # Load and include warp mask from robust_cloth_warp.py if provided
        warp_mask = None
        connection_mask = None
        if warp_mask_path:
            print(f">> Loading warp mask from: {warp_mask_path}")
            warp_mask = cv2.imread(str(warp_mask_path), cv2.IMREAD_GRAYSCALE)
            if warp_mask is not None:
                # Resize to match image dimensions if needed
                if warp_mask.shape[:2] != (h, w):
                    warp_mask = cv2.resize(warp_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                print(f"[OK] Warp mask loaded: {cv2.countNonZero(warp_mask)} pixels")
                
                # CRITICAL CHANGE: If warp_mask is provided (from dresss.py), TRUST IT.
                # Do NOT fill gaps between legs if they exist in warp_mask.
                
                # Combine: warp mask + legs mask
                skin_mask = cv2.bitwise_or(skin_mask, warp_mask)
                
                # OPTIONAL: Connection logic disabled to preserve gaps
                # connection_mask = self._connect_masks_from_below(warp_mask, skin_mask, h, w)
                # skin_mask = cv2.bitwise_or(skin_mask, connection_mask)
                
                print(f"[OK] Combined mask: {cv2.countNonZero(skin_mask)} pixels")
            else:
                print(f"[WARNING] Failed to load warp mask from: {warp_mask_path}")
        
        # Create thick border
        print(f">> Creating thick border (thickness: {border_thickness}px)...")
        border_mask = self.create_thick_border_mask(cloth_mask, border_thickness)
        
        print(f"[OK] Border pixels: {cv2.countNonZero(border_mask)}")
        
        # Combine border mask with exposed skin mask
        combined_mask = cv2.bitwise_or(border_mask, skin_mask)
        
        # AGGRESSIVE hole filling for combined mask - REMOVED
        print(f">> Final mask cleanup (minimal closing)...")
        
        # Step 1: Close small holes only
        kernel_close_small = np.ones((3, 3), np.uint8) # Reduced from 11x11
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close_small)
        
        # Step 2: Medium closing - REMOVED
        # kernel_close_medium = np.ones((21, 21), np.uint8)
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close_medium)
        
        # Step 3: FloodFill - REMOVED
        
        # Step 4: Final closing to smooth edges (very light)
        kernel_final = np.ones((5, 5), np.uint8) # Reduced from 15x15
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_final)
        
        # CRITICAL: Exclude face and hair from the final mask
        print(">> Removing face and hair areas from mask...")
        combined_mask[face_hair_protection > 0] = 0
        print(f"   - Face and hair areas excluded from masking")
        
        print(f"[OK] Combined mask pixels (border + skin): {cv2.countNonZero(combined_mask)}")
        
        # Apply overlay
        print(f">> Applying overlay (color: {overlay_color}, alpha: {alpha})...")
        result = self.apply_overlay(image, combined_mask, overlay_color, alpha)
        
        # Save result
        cv2.imwrite(str(output_path), result)
        print(f"[SAVED] Masked result: {output_path}")
        
        # Also save just the mask for reference
        from pathlib import Path
        output_p = Path(output_path)
        mask_only_path = str(output_p.parent / (output_p.stem + '_mask' + output_p.suffix))
        
        # Prepare the mask to save: cloth border + warp mask + bottom filling + ARMS/HANDS
        mask_to_save = border_mask.copy()
        
        # CRITICAL: Include arms/hands mask (skin_mask) in saved mask
        # This ensures arms covered by red overlay are also in the saved mask
        print(f">> Including arms/hands in saved mask")
        mask_to_save = cv2.bitwise_or(mask_to_save, skin_mask)
        
        # Include warp mask if it was loaded
        if warp_mask is not None:
            print(f">> Including warp mask from robust_cloth_warp.py in saved mask")
            mask_to_save = cv2.bitwise_or(mask_to_save, warp_mask)
        
        # Include bottom filling (connection_mask) if it exists
        if connection_mask is not None:
            # Only include if specifically enabled (currently disabled)
            pass
            # print(f">> Including bottom gap filling in saved mask")
            # mask_to_save = cv2.bitwise_or(mask_to_save, connection_mask)
        
        # Apply minimal clean up
        kernel_close_small = np.ones((3, 3), np.uint8)
        mask_to_save = cv2.morphologyEx(mask_to_save, cv2.MORPH_CLOSE, kernel_close_small)
        
        # REMOVED: Large closing and floodFill which destroys gaps
        # kernel_close_medium = np.ones((21, 21), np.uint8)
        # mask_to_save = cv2.morphologyEx(mask_to_save, cv2.MORPH_CLOSE, kernel_close_medium)
        
        # mask_inv = cv2.bitwise_not(mask_to_save)
        # cv2.floodFill(...)
        # holes = cv2.bitwise_not(mask_inv)
        # mask_to_save = cv2.bitwise_or(mask_to_save, holes)
        
        kernel_final = np.ones((5, 5), np.uint8)
        mask_to_save = cv2.morphologyEx(mask_to_save, cv2.MORPH_CLOSE, kernel_final)
        
        # EXCLUDE MediaPipe legs from the final mask
        if hasattr(self, 'mp_legs_visualization') and self.mp_legs_visualization is not None:
            print(f">> Excluding MediaPipe legs from final saved mask")
            # Remove legs (set pixels to 0 where legs are detected)
            mask_to_save[self.mp_legs_visualization > 0] = 0
            print(f"   - Legs excluded from mask")
        
        # RE-ADD the warp mask from robust_cloth_warp.py at the end
        if warp_mask is not None:
            print(f">> Re-adding warp mask from robust_cloth_warp.py at the end")
            mask_to_save = cv2.bitwise_or(mask_to_save, warp_mask)
        
        # CRITICAL: Exclude face and hair from saved mask too
        print(f">> Removing face and hair areas from saved mask...")
        mask_to_save[face_hair_protection > 0] = 0
        print(f"   - Face and hair areas excluded from saved mask")
        
        cv2.imwrite(mask_only_path, mask_to_save)
        print(f"[SAVED] Mask (with robust_cloth_warp mask included): {mask_only_path}")
        
        # Save LIP segmentation visualization
        if hasattr(self, 'seg_visualization'):
            seg_vis_path = str(output_p.parent / (output_p.stem + '_lip_seg' + output_p.suffix))
            cv2.imwrite(seg_vis_path, self.seg_visualization)
            print(f"[SAVED] LIP segmentation visualization: {seg_vis_path}")
            print("  (Green=Arms, Cyan=Legs, Magenta=Pants, Yellow=Shoes, Blue=Face, Red=Clothes)")
        
        # --- GENERATE SECOND MASK (v2) ---
        # "clotsh adn lover perosn body also masked"
        print(f">> Generating second mask (v2) with clothes and lower body...")
        
        # 1. Start with the original mask
        mask_v2 = mask_to_save.copy()
        
        # 2. Add all cloth regions from SegFormer (cloth_mask)
        mask_v2 = cv2.bitwise_or(mask_v2, cloth_mask)
        
        # 3. Add clothes and lower body from LIP model
        if hasattr(self, 'lip_seg'):
            # LIP Classes: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 9: Left-shoe, 10: Right-shoe, 12: Left-leg, 13: Right-leg
            full_body_classes = [4, 5, 6, 7, 9, 10, 12, 13]
            lip_body_mask = np.isin(self.lip_seg, full_body_classes).astype(np.uint8) * 255
            
            # Dilate slightly to ensure full coverage
            kernel_body = np.ones((5, 5), np.uint8)
            lip_body_mask = cv2.dilate(lip_body_mask, kernel_body, iterations=1)
            
            mask_v2 = cv2.bitwise_or(mask_v2, lip_body_mask)
            print(f"   - Added LIP body/clothes: {cv2.countNonZero(lip_body_mask)} pixels")
            
        # 4. Add MediaPipe legs (which were excluded from mask 1)
        if hasattr(self, 'mp_legs_visualization') and self.mp_legs_visualization is not None:
            mask_v2 = cv2.bitwise_or(mask_v2, self.mp_legs_visualization)
            print(f"   - Added MediaPipe legs: {cv2.countNonZero(self.mp_legs_visualization)} pixels")
            
        # 5. Connect gaps between legs for v2 if needed (optional but recommended for "lower person body")
        # Reuse _connect_masks_from_below but maybe more aggressively
        if warp_mask is not None:
            v2_connection = self._connect_masks_from_below(warp_mask, mask_v2, h, w)
            mask_v2 = cv2.bitwise_or(mask_v2, v2_connection)
            
        # 6. Final cleanup for v2
        kernel_v2 = np.ones((5, 5), np.uint8)
        mask_v2 = cv2.morphologyEx(mask_v2, cv2.MORPH_CLOSE, kernel_v2)
        
        # 7. CRITICAL: Still exclude face and hair from v2
        mask_v2[face_hair_protection > 0] = 0
        
        # 8. Save the second mask
        mask_v2_path = str(output_p.parent / (output_p.stem + '_mask_v2' + output_p.suffix))
        cv2.imwrite(mask_v2_path, mask_v2)
        print(f"[SAVED] Second mask (full body/clothes): {mask_v2_path}")
        
        # Save MediaPipe arms/hands visualization
        if hasattr(self, 'mp_arms_hands_visualization'):
            mp_arms_path = str(output_p.parent / (output_p.stem + '_mp_arms_hands' + output_p.suffix))
            cv2.imwrite(mp_arms_path, self.mp_arms_hands_visualization)
            print(f"[SAVED] MediaPipe arms/hands segmentation: {mp_arms_path}")
            print("  (White=Arms and Hands from MediaPipe)")
        
        # Save MediaPipe legs visualization
        if hasattr(self, 'mp_legs_visualization'):
            mp_legs_path = str(output_p.parent / (output_p.stem + '_mp_legs' + output_p.suffix))
            cv2.imwrite(mp_legs_path, self.mp_legs_visualization)
            print(f"[SAVED] MediaPipe legs segmentation: {mp_legs_path}")
            print("  (White=Legs and Feet from MediaPipe)")
        
        # --- GENERATE THIRD MASK (v3) ---
        if generate_v3:
            # Body parts (hands, legs, feet, shoes) using SAM
            print(f">> Generating third mask (v3) with SAM body parts (Strictly excluding cloth)...")
            
            # 1. Get exclusion and guide masks
            # Exclusion: Both SegFormer (cloth_mask) and B2 (b2_full_cloth)
            b2_full_cloth = np.zeros((h, w), dtype=np.uint8)
            if self.b2_session is not None:
                b2_full_cloth = self.get_b2_cloth_mask(image)
            
            total_cloth_exclusion = cv2.bitwise_or(cloth_mask, b2_full_cloth)
            
            # Guide: Use MediaPipe to define "Safe Zones" for limbs only
            # This will kill any hits on the face or center of torso
            guide_kernel = np.ones((50, 50), np.uint8)
            valid_limb_guide = cv2.dilate(mp_mask, guide_kernel, iterations=1)
            
            # 2. Get body parts using SAM
            # We use a very specific prompt to avoid catching the cloth
            sam_options = SAMOptions(
                dino_prompt='exposed skin on arms, hands, bare legs, feet, shoes',
                dino_box_threshold=0.25,
                dino_text_threshold=0.2,
                max_detections=20
            )
            sam_mask, _, _, _ = generate_mask_from_image(image, mask_model='sam', sam_options=sam_options)
            if sam_mask is not None:
                if len(sam_mask.shape) == 3:
                    sam_mask = sam_mask[:, :, 0]
                
                # CRITICAL: Clean SAM mask using our limb guide
                # This kills face dots and torso artifacts immediately
                sam_mask = cv2.bitwise_and(sam_mask, valid_limb_guide)
                
                # Subtract cloth to be safe
                sam_mask = cv2.subtract(sam_mask, total_cloth_exclusion)
                    
                print(f"   - Guided & Pruned SAM body parts: {cv2.countNonZero(sam_mask)} pixels")
            else:
                sam_mask = np.zeros((h, w), dtype=np.uint8)
                print("   - [WARNING] SAM mask generation failed or returned None")
                
            # 3. Refine and Expand Skin Mask
            # First, strictly remove face and hair from SAM mask BEFORE dilation
            # This prevents the "halo" or "lining" effect around the head
            sam_mask[face_hair_protection > 0] = 0
            
            # Now expand the skin mask (arms, hands, legs, feet)
            # Use 15x15 kernel with 2 iterations for "a little bit more" expansion as requested
            sam_kernel = np.ones((15, 15), np.uint8)
            sam_mask = cv2.dilate(sam_mask, sam_kernel, iterations=2)
            
            # 4. Strictly Remove Cloth and Noise
            mask_v3 = sam_mask.copy()
            
            # Use total cloth for exclusion (dilated for safety)
            cloth_exclusion_final = cv2.dilate(total_cloth_exclusion, np.ones((7, 7), np.uint8), iterations=1)
            mask_v3[cloth_exclusion_final > 0] = 0
            
            # NOISE REMOVAL: Delete small disconnected clusters (face dots, etc.)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_v3, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 200: # Kill small artifacts
                    mask_v3[labels == i] = 0
            
            print(f"   - Final V3 Mask noise cleaned and cloth pruned")
            
            # 5. Final Protection Pass (Strict removal of face/hair to fix lining)
            # Subtract dilated protection zone from the FINAL combined mask
            protection_v3 = cv2.dilate(face_hair_protection, np.ones((25, 25), np.uint8), iterations=2)
            mask_v3[protection_v3 > 0] = 0
            
            # 6. Save the third mask
            mask_v3_path = str(output_p.parent / (output_p.stem + '_mask_v3' + output_p.suffix))
            cv2.imwrite(mask_v3_path, mask_v3)
            print(f"[SAVED] Third mask (Strict Anatomy Only): {mask_v3_path}")
        
        print(f"{'='*60}\n")


# ============== MAIN ==============
if __name__ == "__main__":
    # Initialize masker with SegFormer b3-fashion model and LIP parsing model
    masker = ClothMasker(
        model_path=r"D:\Models\segformer-b3-fashion",
        onnx_model_path=r"D:\IDM_VTON_Weights\ckpt\humanparsing\parsing_lip.onnx",
        b2_onnx_path=r"D:\IDM_VTON_Weights\SegFormerB2Clothes\segformer_b2_clothes.onnx"
    )
    
    # Process the result image from robust_cloth_warp.py
    # Adjust the input path to your actual output from robust_cloth_warp.py
    input_image = r"C:\Users\PC\.gemini\antigravity\scratch\output1\FINAL_DRESS12.png"
    output_image = r"C:\Users\PC\Downloads\image (24)_masked.png"
    
    # Optional: Path to the _MASK.png file produced by robust_cloth_warp.py
    # This will include the masked areas (clothes, arms, hands, legs, shoes) from the warp process
    warp_mask_image = r"C:\Users\PC\.gemini\antigravity\scratch\output1\FINAL_DRESS12_MASK.png"
    
    masker.process(
        input_image_path=input_image,
        output_path=output_image,
        border_thickness=15,  # Thicker border extending outward
        overlay_color=(0, 0, 255),  # BGR: Red color for outline
        alpha=0.5,  # Overlay transparency (0.0 = invisible, 1.0 = solid)
        warp_mask_path=warp_mask_image  # Include masked areas from robust_cloth_warp.py
    )
    
    print("\n[COMPLETE] Masking complete!")
    print("\nTo use with different images, modify the paths in __main__ section:")
    print("  - input_image: Path to result from robust_cloth_warp.py")
    print("  - output_image: Where to save the masked result")
    print("  - warp_mask_image: Path to _MASK.png from robust_cloth_warp.py (optional)")
    print("  - border_thickness: Adjust for thicker/thinner outline")
    print("  - overlay_color: Change color (BGR format)") 
    print("  - alpha: Change transparency (0.0-1.0)")

