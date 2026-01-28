import cv2
import numpy as np
import torch
import mediapipe as mp

class VTONMasker:
    """
    Unified VTON Masking Helper.
    Generates high-quality masks for Inpainting/Composite blending.
    Logic ported from 'dresss - Copy (2).py' to standardise protection & removal.
    """
    
    def __init__(self, seg_model_path=None, b3_model_path=None, sam_model_path=None):
        from .model_loader import get_b2_session, get_b3_model_and_processor, get_sam_predictor, get_pose_detector, get_device
        
        self.device = get_device()
        
        # 1. B2 (ONNX)
        self.session = get_b2_session(seg_model_path) if seg_model_path else None
        self.input_name = self.session.get_inputs()[0].name if self.session else None
        
        # 2. B3 (Fashion)
        self.model, self.processor = (None, None)
        if b3_model_path:
            self.model, self.processor = get_b3_model_and_processor(b3_model_path)
            
        # 3. MediaPipe
        self.pose_detector = get_pose_detector()
        
        # 4. SAM
        self.predictor = get_sam_predictor(sam_model_path) if sam_model_path else None

    def _segment_b3(self, image):
        """Runs SegFormer B3 (High Quality). Returns class map."""
        if self.model is None: return None
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def _segment_b2(self, image):
        """Runs SegFormer B2 (ONNX). Returns class map."""
        if self.session is None: return None
        h, w = image.shape[:2]
        
        # B2 Preprocessing
        img_resized = cv2.resize(image, (512, 512))
        img_norm = (cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        img_tensor = img_norm.transpose(2,0,1)[None].astype(np.float32)
        
        outputs = self.session.run(None, {self.input_name: img_tensor})
        logits = outputs[0][0] # [18, 512, 512]
        
        # Optimized: Perform argmax at low-res THEN resize labels to avoid massive memory usage on 4K+ images
        seg_small = np.argmax(logits, axis=0).astype(np.uint8)
        return cv2.resize(seg_small, (w, h), interpolation=cv2.INTER_NEAREST)

    def get_seg_map(self, image):
        """Prefer B3 if available, else B2."""
        if self.model: return self._segment_b3(image)
        return self._segment_b2(image)

    def get_base_mask(self, image, mode='universal'):
        """
        Generates the Aggressive "Base Mask" of parts to remove.
        Refined logic from 'dresss - Copy (2).py'.
        """
        seg = self.get_seg_map(image)
        if seg is None: return np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        
        # --- Model Detection ---
        is_b3 = np.max(np.unique(seg)) > 18
        
        if is_b3:
            # --- B3 FASHION ONLY ---
            UPPER_CLOTH = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 28, 29, 32, 34, 36} | set(range(37, 47))
            LOWER_CLOTH = {7, 8, 9, 20}
            SHOES = {24}
            # FOR B3, WE ALWAYS FETCH B2 FOR BODY KNOWLEDGE
            seg_b2 = self._segment_b2(image)
            HEAD_ACC = {1, 3} # Hat, Glasses (B2)
            BODY_SAFE = {2, 11} # Hair, Face (B2)
            ARMS = {14, 15}
            LEGS = {12, 13}
            # Re-map B3 Leg accessories
            LEGS_ACC = {21, 22, 23} 
        else:
            # --- B2 FULL BODY ---
            UPPER_CLOTH = {4, 7, 16, 17}
            LOWER_CLOTH = {5, 6, 8}
            SHOES = {9, 10}
            HEAD_ACC = {1, 3} # Hat, Glasses
            BODY_SAFE = {2, 11} # Hair, Face
            ARMS = {14, 15}
            LEGS = {12, 13}
            LEGS_ACC = set()

        # ... (rest of logic) ...
        remove_mask = np.zeros((h, w), dtype=np.uint8)
        protect_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use B2 segment if B3 was active for body knowledge
        active_seg = seg if not is_b3 else seg_b2

        if mode == 'universal' or mode == 'dress':
            targets = UPPER_CLOTH | LOWER_CLOTH | LEGS
            protected = BODY_SAFE | HEAD_ACC | SHOES
        elif mode == 'upper' or mode == 'shirt':
            # USER REQUEST: Connect sleeves to hands. 
            # To do this, we MUST remove the original arms (long sleeves) so the inpainter can fill them.
            targets = UPPER_CLOTH | ARMS 
            protected = BODY_SAFE | HEAD_ACC | LOWER_CLOTH | SHOES | LEGS
        elif mode == 'lower' or mode == 'pants':
            targets = LOWER_CLOTH | LEGS
            protected = BODY_SAFE | HEAD_ACC | UPPER_CLOTH | ARMS | SHOES
        else:
            return np.zeros((h, w), dtype=np.uint8)
            
        # Garment targets come from لباس map
        for c in targets: remove_mask[seg == c] = 255
        # Protection comes from body knowledge
        for c in protected: protect_mask[active_seg == c] = 255
            
        # Refinement
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        remove_mask = cv2.dilate(remove_mask, k, iterations=4) # Increased from 2 for better removal
        protect_mask = cv2.dilate(protect_mask, k, iterations=1)
        
        # --- HAND & FEET PROTECTION ---
        # For universal/dress, protect feet too. For others, just hands.
        protect_feet = (mode in ['universal', 'dress'])
        hand_prot = self.get_mediapipe_mask(image, hands_only=(not protect_feet))
        protect_mask = cv2.bitwise_or(protect_mask, hand_prot)

        remove_mask[protect_mask > 0] = 0
        remove_mask = cv2.morphologyEx(remove_mask, cv2.MORPH_OPEN, k)
        remove_mask = cv2.morphologyEx(remove_mask, cv2.MORPH_CLOSE, k)
        
        return remove_mask

    def get_extremities_mask(self, image):
        """
        Extracts Hands and Legs (exposed skin/limbs).
        """
        seg = self.get_seg_map(image)
        if seg is None: return np.zeros(image.shape[:2], dtype=np.uint8)
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        EXTREMITIES_CLASSES = {12, 13, 9, 10} # Legs, Shoes (Excluded Arms 14, 15)
        
        for c in EXTREMITIES_CLASSES:
            mask[seg == c] = 255
            
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=1)
        
        return mask

    def get_diff_mask(self, original, clean, threshold=10):
        """
        Calculates the mask of 'Blurred' or 'Changed' areas.
        """
        if original is None or clean is None: return None
        if original.shape != clean.shape:
            clean = cv2.resize(clean, (original.shape[1], original.shape[0]))
            
        diff = cv2.absdiff(original, clean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, k)
        diff_mask = cv2.dilate(diff_mask, k, iterations=5) # Increased from 2 for aggressive removal area coverage
        
        return diff_mask

    def get_mediapipe_mask(self, image, hands_only=False):
        """
        Generates robust mask for Hands and Feet using MediaPipe Landmarks.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(img_rgb)
        
        if not results.pose_landmarks:
            return mask
            
        landmarks = results.pose_landmarks.landmark
        base_size = min(h, w)
        # --- TUNED: Expanded radius for full finger/wrist coverage ---
        hand_radius = int(base_size * 0.050) 
        foot_radius = int(base_size * 0.055)
        limb_thickness = int(base_size * 0.050)
        
        LEFT_HAND = [15, 17, 19, 21]
        RIGHT_HAND = [16, 18, 20, 22]
        LEFT_FOOT = [27, 29, 31]
        RIGHT_FOOT = [28, 30, 32]
        
        for idx in LEFT_HAND + RIGHT_HAND:
            if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                cv2.circle(mask, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), hand_radius, 255, -1)
                
        if not hands_only:
            for idx in LEFT_FOOT + RIGHT_FOOT:
                if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                    cv2.circle(mask, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), foot_radius, 255, -1)
                
        # Connections
        if hands_only:
            connections = [(15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)]
        else:
            connections = [(15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (27, 29), (27, 31), (28, 30), (28, 32)]

        for start, end in connections:
             if landmarks[start].visibility > 0.5 and landmarks[end].visibility > 0.5:
                  pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
                  pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
                  cv2.line(mask, pt1, pt2, 255, limb_thickness)
        return mask

    def get_upper_body_mask(self, image):
        """Extracts all upper-body clothing classes as a single dilated mask."""
        seg = self.get_seg_map(image)
        if seg is None: return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Determine if B3 was used (classes > 18)
        is_b3 = np.max(np.unique(seg)) > 18
        
        if is_b3:
            # UPPER_CLOTH classes for B3
            UPPER_CLOTH = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 28, 29, 32, 34, 36} | set(range(37, 47))
        else:
            # UPPER_CLOTH classes for B2
            UPPER_CLOTH = {4, 7, 16, 17}
            
        # Create mask
        upper_mask = np.isin(seg, list(UPPER_CLOTH)).astype(np.uint8) * 255
        
        # Small dilation to ensure clean coverage
        if np.any(upper_mask > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            upper_mask = cv2.dilate(upper_mask, kernel, iterations=1)
            
        return upper_mask

    def get_precise_hand_mask(self, image):
        """
        Combines SegFormer-B2's body shape with MediaPipe's proximity guide.
        Note: We force B2 here because B3 lacks skin/arm labels.
        """
        h, w = image.shape[:2]
        
        # 1. Get Proper Body Shape (Always B2)
        seg_body = self._segment_b2(image)
        if seg_body is None: 
            return self.get_mediapipe_mask(image, hands_only=True) # Fallback
            
        # Labels 14, 15 are Left/Right Arms in B2
        seg_hand_mask = np.isin(seg_body, [14, 15]).astype(np.uint8) * 255
        
        # Dilate SegFormer base to ensure no "inner" edges show cloth
        k_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seg_hand_mask = cv2.dilate(seg_hand_mask, k_base, iterations=1)
        
        # 2. Get Proximity Guide (MediaPipe)
        mp_hand_mask = self.get_mediapipe_mask(image, hands_only=True)
        
        # 3. Intersect (Keep SegFormer shape but only where MP says a hand is)
        precise_mask = cv2.bitwise_and(seg_hand_mask, mp_hand_mask)
        
        # 4. Final Safety Expansion for fingertips
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        precise_mask = cv2.dilate(precise_mask, kernel, iterations=2)
        
        return precise_mask

    def get_cloth_edge_mask(self, warped_alpha, thickness=3):
        if warped_alpha is None: return None
        _, binary = cv2.threshold(warped_alpha, 127, 255, cv2.THRESH_BINARY)
        k = np.ones((thickness, thickness), np.uint8)
        edge_mask = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, k)
        return edge_mask

    def get_face_shield(self, original_img):
        """
        Generates a robust Face & Hair protection mask.
        Combines SegFormer and SAM (if available) with strict chin protection.
        """
        h, w = original_img.shape[:2]
        face_shield = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Base SegFormer Protection (Face=11, Hair=2)
        seg_b2 = self._segment_b2(original_img)
        if seg_b2 is not None:
             base_shield = np.isin(seg_b2, [2, 11]).astype(np.uint8) * 255
             face_shield = cv2.bitwise_or(face_shield, base_shield)
        
        # 2. SAM Precise Protection (If available)
        if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
             img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
             res = self.pose_detector.process(img_rgb)
             if res.pose_landmarks:
                  nose = res.pose_landmarks.landmark[0]
                  input_point = np.array([[int(nose.x * w), int(nose.y * h)]])
                  self.sam_predictor.set_image(img_rgb)
                  masks, _, _ = self.sam_predictor.predict(input_point, np.array([1]), multimask_output=True)
                  sam_head = masks[np.argmax([m.sum() for m in masks])].astype(np.uint8) * 255
                  face_shield = cv2.bitwise_or(face_shield, sam_head)

        # 3. Ultra-Safe Chin Capping
        # Relaxed to 0.60 to cover full chin/jawline
        res = self.pose_detector.process(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
             lm = res.pose_landmarks.landmark
             mouth_y = (lm[9].y + lm[10].y) / 2
             nose_y = lm[0].y
             face_height = max(1e-3, mouth_y - nose_y)
             chin_limit_y = int((mouth_y + face_height * 0.60) * h)
             
             cap_mask = np.zeros_like(face_shield)
             cap_mask[:chin_limit_y, :] = 255
             face_shield = cv2.bitwise_and(face_shield, cap_mask)

        # 4. Dilate for buffer
        k_shield = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        face_shield = cv2.dilate(face_shield, k_shield, iterations=1)
        
        return face_shield

    def get_final_mask(self, original_img, warped_mask, mode='universal', clean_img=None, garment_type=None):
        """
        Generates the UNIFIED Mask.
        - mode='shirt': PANTS-STYLE LOGIC (Base + Edges only, No Diff).
        - mode='universal': Includes Diff-mask for complex changes.
        """
        h, w = original_img.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. BASE REMOVAL MASK (Area where the old cloth was)
        print(f"  🎭 VTONMasker: Generating Base Removal Mask (Mode: {mode})...")
        base_mask = self.get_base_mask(original_img, mode)
        final_mask = cv2.bitwise_or(final_mask, base_mask)

        # 2. DIFF MASK (Crucial for masking the blurred "Removed Cloth" area)
        # Previously restricted to non-shirt modes; now enabled for all when clean_img is available.
        if clean_img is not None:
                print("  🎭 VTONMasker: Calculating Diff Mask (Blurred Areas)...")
                diff_mask = self.get_diff_mask(original_img, clean_img)
                if diff_mask is not None:
                    final_mask = cv2.bitwise_or(final_mask, diff_mask)

        # 3. REGIONAL PROTECTION GUARD
        # Force-shield the other half of the person
        seg = self.get_seg_map(original_img)
        if seg is not None:
             is_b3 = np.max(np.unique(seg)) > 18
             
             if is_b3:
                  # Shield only what we ARE SURE is not a shirt or skin
                  LOWER = {7, 8, 9, 20} # Pants, Shorts, Skirt, Belt
                  HEAD = {14, 15} # Glasses, Hat (Label 16 is hair-acc/hoodie-ish, unshield it)
                  SHOES = {24}
             else:
                  LOWER = {5, 6, 8, 9, 10, 12, 13}
                  UPPER = {4, 7, 16, 17}
                  HEAD = {1, 2, 3, 11}
             
             guard_mask = np.zeros_like(final_mask)
             if mode in ['shirt', 'upper']:
                  print("  🎭 VTONMasker: Shielding Lower Body & Head from Masking...")
                  guard_indices = (LOWER | HEAD | SHOES) if is_b3 else (LOWER | HEAD)
                  for c in guard_indices:
                       guard_mask[seg == c] = 255
             elif mode in ['pants', 'lower']:
                  print("  🎭 VTONMasker: Shielding Upper Body & Head from Masking...")
                  guard_indices = UPPER if is_b3 else (UPPER | HEAD)
                  for c in guard_indices:
                       guard_mask[seg == c] = 255
             
             if np.any(guard_mask > 0):
                  k_guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                  guard_mask = cv2.dilate(guard_mask, k_guard, iterations=1)
                  final_mask[guard_mask > 0] = 0

        # 4. WARPED CLOTH HANDLING (Edge Expansion)
        if warped_mask is not None:
            if warped_mask.ndim == 3: warped_mask = warped_mask[:, :, 0]
            if warped_mask.shape[:2] != (h, w):
                warped_mask = cv2.resize(warped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            print("  🎭 VTONMasker: Adding Warped Cloth Boundary Edges...")
            _, cloth_binary = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Edge Blending (Corners/Edges of the NEW cloth)
            # Increase thickness to mask more outside/inside as requested
            edge_thickness = 45 if mode in ['pants', 'lower'] else 35 
            edge_mask = self.get_cloth_edge_mask(warped_mask, thickness=edge_thickness)
            final_mask = cv2.bitwise_or(final_mask, edge_mask)
            
            # 4b. HORIZONTAL EDGE EXPANSION (User Request: "expand left/right sides")
            if mode in ['shirt', 'upper']:
                 # Horizontal-biased dilation to stretch mask outward on sides
                 k_horiz = np.ones((5, 45), np.uint8)
                 horiz_edge = cv2.dilate(edge_mask, k_horiz, iterations=1)
                 final_mask = cv2.bitwise_or(final_mask, horiz_edge)
            elif mode in ['pants', 'lower']:
                 # USER REQUEST: Expand mask from sides for pants too.
                 print("  🎭 VTONMasker: Applying Horizontal Side Expansion for Pants...")
                 k_horiz_pants = np.ones((1, 45), np.uint8) # Strictly horizontal
                 side_edge = cv2.dilate(edge_mask, k_horiz_pants, iterations=1)
                 final_mask = cv2.bitwise_or(final_mask, side_edge)
            elif mode in ['universal', 'dress']:
                 # USER REQUEST: Significant Horizontal Expansion for Universal
                 print("  🎭 VTONMasker: Applying Wide Horizontal Expansion for Universal...")
                 k_horiz_univ = np.ones((1, 65), np.uint8) # Very wide horizontal kernel
                 univ_side_edge = cv2.dilate(edge_mask, k_horiz_univ, iterations=1)
                 final_mask = cv2.bitwise_or(final_mask, univ_side_edge)

        # 5. HAND MASKING (User Request: "also mask the person hands" - Precise Shape Style)
        if mode in ['shirt', 'upper']:
            print("  🎭 VTONMasker: Adding Precise Hand Masking (SegFormer + MediaPipe)...")
            hand_mask = self.get_precise_hand_mask(original_img)
            final_mask = cv2.bitwise_or(final_mask, hand_mask)

        # 6. HOODIE / COLLAR / UPPER EXPANSION (User Request)
        # Targeted Neck Zone Expansion (Vertical Bias)
        if mode in ['shirt', 'upper', 'universal']:
             neck_zone_mask = np.zeros_like(final_mask)
             # Use MediaPipe landmarks to define neck/hood area
             img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
             res = self.pose_detector.process(img_rgb)
             if res.pose_landmarks:
                  lm = res.pose_landmarks.landmark
                  # Neck region: between shoulders and nose
                  sh_y = (lm[11].y + lm[12].y) / 2
                  nose_y = lm[0].y
                  top = int(max(0, (nose_y - 0.05) * h))
                  bottom = int(min(h, (sh_y + 0.05) * h))
                  left = int(max(0, (lm[12].x - 0.1) * w))
                  right = int(min(w, (lm[11].x + 0.1) * w))
                  neck_zone_mask[top:bottom, left:right] = 255
             else:
                  # Fallback to general upper 30%
                  neck_zone_mask[:int(h*0.3), :] = 255
             
             upper_cloth_parts = cv2.bitwise_and(final_mask, neck_zone_mask)
             if np.any(upper_cloth_parts > 0):
                  print("  🎭 VTONMasker: Applying Vertical Neckline/Hood Expansion...")
                  # Apply vertical-bias expansion for collars/hoods
                  k_neck = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 65))
                  neck_expansion = cv2.dilate(upper_cloth_parts, k_neck, iterations=1)
                  final_mask = cv2.bitwise_or(final_mask, neck_expansion)

        # 6b. SLEEVE & BOTTOM DOWNWARD EXPANSION (User Request)
        if warped_mask is not None and mode in ['shirt', 'upper']:
            print("  🎭 VTONMasker: Expanding Sleeve and Bottom edges downwards...")
            _, cloth_bin = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find bottom pixels of the entire cloth
            cloth_coords = np.where(cloth_bin > 0)
            if len(cloth_coords[0]) > 0:
                y_max = np.max(cloth_coords[0])
                # Expand downward at the very bottom
                bottom_strip = np.zeros_like(final_mask)
                bottom_strip[max(0, y_max-55):min(h, y_max+10), :] = 255

                bottom_active = cv2.bitwise_and(cloth_bin, bottom_strip)
                k_down = np.ones((65, 7), np.uint8)
                bottom_expanded = cv2.dilate(bottom_active, k_down, iterations=3)

                final_mask = cv2.bitwise_or(final_mask, bottom_expanded)

            # Find sleeve bottoms using landmarks if available
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            res = self.pose_detector.process(img_rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                # Wrist landmarks
                for idx in [15, 16]:
                    if lms[idx].visibility > 0.5:
                        wx, wy = int(lms[idx].x * w), int(lms[idx].y * h)
                        # Create a small area around wrist and expand downward
                        sleeve_bottom_zone = np.zeros_like(final_mask)
                        cv2.circle(sleeve_bottom_zone, (wx, wy), int(w * 0.05), 255, -1)
                        sleeve_active = cv2.bitwise_and(cloth_bin, sleeve_bottom_zone)
                        if np.any(sleeve_active > 0):
                            k_sleeve = np.ones((25, 15), np.uint8)
                            sleeve_expanded = cv2.dilate(sleeve_active, k_sleeve, iterations=2)
                            final_mask = cv2.bitwise_or(final_mask, sleeve_expanded)

        # 6c. LEG GAP FILLER (User Request: "Fill gap between legs")
        if mode in ['universal', 'dress']:
             print("  🎭 VTONMasker: Filling Leg Gap (Solidifying Lower Body)...")
             # Focus on the lower 45% of the image (Legs area only)
             leg_zone_y = int(h * 0.55)
             leg_zone = final_mask[leg_zone_y:, :]
             
             if np.any(leg_zone > 0):
                  # Very wide horizontal kernel to bridge the gap between legs
                  # Height 15, Width 120 (Strong Horizontal Bridge)
                  k_bridge = np.ones((15, 120), np.uint8)
                  
                  # Morphological CLOSE to fill the gap between two white regions (legs)
                  filled_legs = cv2.morphologyEx(leg_zone, cv2.MORPH_CLOSE, k_bridge)
                  
                  # Integrate back into final mask
                  final_mask[leg_zone_y:, :] = cv2.bitwise_or(final_mask[leg_zone_y:, :], filled_legs)


        # 7. SOLID ARM BRIDGES (User Request: "Solid bridge between sleeves and hands")
        if mode in ['shirt', 'upper', 'universal']:
             print("  🎭 VTONMasker: Building Solid Arm Bridges using Segmentation...")
             seg_b2 = self._segment_b2(original_img)
             if seg_b2 is not None:
                  # ARM Segments: Left Arm (14), Right Arm (15)
                  arm_mask = np.isin(seg_b2, [14, 15]).astype(np.uint8) * 255
                  # Dilate significantly to bridge gaps between sleeves and hands
                  k_arm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
                  arm_mask = cv2.dilate(arm_mask, k_arm, iterations=1)
                  # Add to final mask
                  final_mask = cv2.bitwise_or(final_mask, arm_mask)

        # 7.5. ABSOLUTE INTERIOR PROTECTION (User Request: "No masking inside the cloth")
        # We perform this here to ensure we 'carve out' the garment body from all expansions/base masks.
        if mode in ['shirt', 'upper', 'pants', 'lower', 'universal', 'dress'] and warped_mask is not None:
             print("  🎭 VTONMasker: Performing Solid Interior Carve-Out (with Neck Exclusion)...")
             _, cloth_bin = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
             
             # Fill all internal holes in the cloth binary to ensure solid protection
             contours, _ = cv2.findContours(cloth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             cv2.drawContours(cloth_bin, contours, -1, 255, -1)
             
             cv2.drawContours(cloth_bin, contours, -1, 255, -1)
             
             # Use a smaller erosion to ensure sleeves are also caught.
             # USER REQUEST: For skirts, erode MORE (expand mask inward).
             if garment_type == 'skirt':
                 print("  🎭 VTONMasker: Using Expanded Inward Masking for Skirt...")
                 k_inner = np.ones((51, 51), np.uint8)
             else:
                 k_inner = np.ones((31, 31), np.uint8)
                 
             cloth_interior = cv2.erode(cloth_bin, k_inner, iterations=1)
             
             # Neck Exclusion (as before)
             img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
             res = self.pose_detector.process(img_rgb)
             if res.pose_landmarks:
                  lm = res.pose_landmarks.landmark
                  sh_y = (lm[11].y + lm[12].y) / 2
                  neck_limit_y = int((sh_y + 0.02) * h)
                  cloth_interior[:neck_limit_y, :] = 0

             # Force zero masking in the protected interior
             final_mask[cloth_interior > 0] = 0

        # 8. Smoothing
        final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)






                # === FINAL GUARANTEE FIX: FULL SLEEVE BOTTOM 30% MASK (NO POSE, NO FAILURES) ===
        if warped_mask is not None and mode in ['shirt', 'upper']:
            print("  🎭 Fixing sleeves: full-width bottom 30% masking (guaranteed)...")

            _, cloth_bin = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
            h, w = cloth_bin.shape

            ys, xs = np.where(cloth_bin > 0)
            if len(xs) > 0:
                x_min, x_max = np.min(xs), np.max(xs)
                x_width = x_max - x_min

                # Define sleeve zones by horizontal extremes
                left_zone  = (xs < x_min + int(x_width * 0.28))
                right_zone = (xs > x_max - int(x_width * 0.28))

                for zone_idx, zone_mask in enumerate([left_zone, right_zone]):
                    zy = ys[zone_mask]
                    zx = xs[zone_mask]

                    if len(zy) < 100:
                        continue

                    y_top = np.min(zy)
                    y_bottom = np.max(zy)
                    sleeve_height = y_bottom - y_top

                    if sleeve_height < 20:
                        continue

                    # USER REQUEST: Bottom 30% of FULL sleeve
                    mask_h = int(sleeve_height * 0.30)

                    sleeve_bottom = np.zeros_like(cloth_bin)
                    sleeve_bottom[
                        y_bottom - mask_h : y_bottom,
                        :
                    ] = 255

                    sleeve_pixels = np.zeros_like(cloth_bin)
                    sleeve_pixels[zy, zx] = 255

                    sleeve_final = cv2.bitwise_and(sleeve_pixels, sleeve_bottom)

                    # --- USER REQUEST: INNER HORIZONTAL EXPANSION ---
                    # Specifically expand the mask horizontally towards the torso from the inner side.
                    # Kernel width of 41 corresponds to the skeleton dilation used elsewhere.
                    k_in = np.ones((1, 41), np.uint8)
                    if zone_idx == 0: # Left Zone (on left side of image) -> Expand RIGHT
                        sleeve_inner = cv2.dilate(sleeve_pixels, k_in, anchor=(0, 0))
                    else: # Right Zone -> Expand LEFT
                        sleeve_inner = cv2.dilate(sleeve_pixels, k_in, anchor=(40, 0))
                    
                    # Merge expansions
                    sleeve_final = cv2.bitwise_or(sleeve_final, sleeve_inner)

                    # Mild smoothing to avoid jagged edges
                    k_smooth = np.ones((10, 5), np.uint8)
                    sleeve_final = cv2.dilate(sleeve_final, k_smooth, iterations=1)

                    final_mask = cv2.bitwise_or(final_mask, sleeve_final)

        # === FINAL GUARANTEE FIX: PANTS BOTTOM 30% MASK (User Request) ===
        if warped_mask is not None and mode in ['pants', 'lower']:
            print("  🎭 Fixing pants: full-width bottom 30% masking (guaranteed)...")
            _, cloth_bin = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
            
            ys, xs = np.where(cloth_bin > 0)
            if len(ys) > 0:
                y_min, y_max = np.min(ys), np.max(ys)
                total_height = y_max - y_min
                
                if total_height > 50:
                    # Target the bottom 30% of the pants garment
                    mask_h = int(total_height * 0.30)
                    
                    bottom_zone = np.zeros_like(cloth_bin)
                    bottom_zone[y_max - mask_h : y_max, :] = 255
                    
                    # Mask only WHERE THE CLOTH IS in that bottom 30% region
                    pants_bottom_mask = cv2.bitwise_and(cloth_bin, bottom_zone)
                    
                    # Dilate vertically to ensure clean coverage of shoes/ankles
                    k_pants = np.ones((41, 15), np.uint8)
                    pants_expanded = cv2.dilate(pants_bottom_mask, k_pants, iterations=1)
                    
                    final_mask = cv2.bitwise_or(final_mask, pants_expanded)



        # 9. MAXIMUM ABSOLUTE FACE & HAIR PROTECTION (Moved to END for Override safety)
        print("  🎭 VTONMasker: Applying SAM-Backed Absolute Face & Hair Shield (Final Pass)...")
        
        # 9a. Base SegFormer Protection
        seg_b2 = self._segment_b2(original_img)
        head_shield = np.zeros_like(final_mask)
        if seg_b2 is not None:
             head_shield = np.isin(seg_b2, [2, 11]).astype(np.uint8) * 255
        
        # 9b. SAM Precise Protection (If available)
        if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
             img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
             res = self.pose_detector.process(img_rgb)
             if res.pose_landmarks:
                  nose = res.pose_landmarks.landmark[0]
                  input_point = np.array([[int(nose.x * w), int(nose.y * h)]])
                  # Predict
                  self.sam_predictor.set_image(img_rgb)
                  masks, _, _ = self.sam_predictor.predict(input_point, np.array([1]), multimask_output=True)
                  sam_head = masks[np.argmax([m.sum() for m in masks])].astype(np.uint8) * 255
                  head_shield = cv2.bitwise_or(head_shield, sam_head)

        # 9c. Ultra-Safe Chin Capping
        res = self.pose_detector.process(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
             lm = res.pose_landmarks.landmark
             mouth_y = (lm[9].y + lm[10].y) / 2
             nose_y = lm[0].y
             face_height = max(1e-3, mouth_y - nose_y)
             # RELAXED: 0.05 -> 0.60 to cover full chin properly
             chin_limit_y = int((mouth_y + face_height * 0.60) * h)
             
             cap_mask = np.zeros_like(final_mask)
             cap_mask[:chin_limit_y, :] = 255
             head_shield = cv2.bitwise_and(head_shield, cap_mask)

        # Apply a 41px exclusion buffer
        k_shield = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        face_shield = cv2.dilate(head_shield, k_shield, iterations=1)
        final_mask[face_shield > 0] = 0
        
        return final_mask
