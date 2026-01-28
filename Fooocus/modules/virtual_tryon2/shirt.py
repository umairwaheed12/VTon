import os
# FORCE Python implementation of Protobuf to avoid MediaPipe GetPrototype error
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path
from modules.virtual_tryon2.vton_masking_helper import VTONMasker
from modules.virtual_tryon2.artificial_skin_helper import ArtificialSkinHelper

class FixedShirtPantsWarper:
    """
    Fixed version with proper cutting and visualization
    """
    
    def __init__(self, b2_model_path, b3_model_path):
        # B2 ONNX (Legacy/Fallback)
        self.session = ort.InferenceSession(str(b2_model_path))
        self.input_name = self.session.get_inputs()[0].name
        
        # B3 Torch (Primary for Fashion)
        print(f"Loading SegFormer-B3 model from: {b3_model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.b3_processor = SegformerImageProcessor.from_pretrained(b3_model_path)
        self.b3_model = SegformerForSemanticSegmentation.from_pretrained(b3_model_path).to(self.device)
        self.b3_model.eval()
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Discover project root (2 levels up from Modules/Virtual_Tryon2 finds Fooocus root)
        self.base_dir = Path(__file__).resolve().parents[2]
        
        # Unified Masking Helper
        sam_path = self.base_dir / "models" / "sam" / "sam_vit_b_01ec64.pth"
        self.masker = VTONMasker(
            seg_model_path=str(b2_model_path), 
            b3_model_path=str(b3_model_path),
            sam_model_path=str(sam_path) if sam_path.exists() else None
        )
        
        # Artificial Skin Helper
        self.skin_helper = ArtificialSkinHelper()
    
    def segment_cloth_b3(self, image):
        """Segment using SegFormer-B3 with higher resolution and fashion labels"""
        H, W = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process and Inference
        inputs = self.b3_processor(images=image_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.b3_model(**inputs)
            logits = outputs.logits
            
        # Resize logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # Get mask
        mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return mask
    
    def segment_cloth(self, image):
        """Segment using SegFormer (matching segformer_legend.py logic)"""
        # Preprocess
        img_resized = cv2.resize(image, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_rgb / 255.0 - mean) / std
        img_tensor = img_normalized.transpose(2, 0, 1).astype(np.float32)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img_tensor})
        
        # Post-process (Fix for zigzag edges: Upscale LOGITS with bilinear interpolation)
        logits = outputs[0][0] # (C, H_s, W_s)
        logits = logits.transpose(1, 2, 0) # (H_s, W_s, C)
        
        # Resize logits to full resolution
        logits_highres = cv2.resize(logits, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Argmax on high-res logits
        seg_map = np.argmax(logits_highres, axis=2).astype(np.uint8)
        
        return seg_map
    
    def draw_labels(self, image, seg_map):
        """Draws class index numbers at the centroid of each segment"""
        vis = image.copy()
        uniques = np.unique(seg_map)
        for u in uniques:
            if u == 0: continue # Skip background
            
            # Find centroid
            mask = (seg_map == u).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 200: continue # Skip massive noise or tiny bits
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Draw text with outline for readability
                    text = str(u)
                    # White border
                    cv2.putText(vis, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3) 
                    # Black text
                    cv2.putText(vis, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        return vis

    def extract_and_save_clothing(self, cloth_img, seg_map, output_dir):
        """
        Extract ONLY clothing as requested:
        Using B3 labels for comprehensive shirt detection.
        """
        print("\n[Step 1] Extracting clothing from segmentation...")
        
        H, W = seg_map.shape
        unique_classes = np.unique(seg_map)
        print(f"  Detected classes: {unique_classes.tolist()}")
        
        # B3 Comprehensive Shirt Group:
        # Labels identified: 1-6 (tops), 10 (coat), 11 (dress), etc.
        # EXCLUDED: 7 (pants), 8 (shorts), 9 (skirt)
        shirt_classes = [1, 2, 3, 4, 5, 6, 10, 11, 13, 17, 20, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
        
        # Debug Visualization colors (Simplified for B3)
        # Using a mix of colors for the main types
        colors = {
            1: [170, 0, 51],    # shirt, blouse
            2: [170, 0, 51],    # top, t-shirt
            32: [255, 0, 0],    # sleeve
            34: [0, 255, 0],    # neckline
        }
        
        # Create masks
        shirt_mask = np.isin(seg_map, shirt_classes).astype(np.uint8) * 255
            
        # Clean masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel)
        shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_OPEN, kernel)
        
        print(f"  Shirt pixels (B3 Combined): {np.sum(shirt_mask > 0)}")

        # --- USER REQUEST: TRIM INNER NECK/LABEL AREA ---
        NECK_LABELS = [28, 29, 34] # Hood, Collar, Neckline
        neck_mask = np.isin(seg_map, NECK_LABELS).astype(np.uint8) * 255
        
        trim_center = None
        trim_axes = None
        
        is_coat = (10 in unique_classes)
        if is_coat:
            print("  🧥 Detected Coat for shallow trimming rule.")
            
        if np.sum(neck_mask > 0) > 0:
            # Condition A: Neck features found. Target them specifically.
            nx, ny, nw, nh = cv2.boundingRect(cv2.findNonZero(neck_mask))
            print(f"  ✂ Neck/Collar detected for trimming at: y={ny}")
            trim_center = (nx + nw // 2, ny)
            # Size proportional to the neck feature width
            # USER REQUEST: SHALLOWER FOR COAT
            if is_coat:
                trim_axes = (int(nw * 0.25), int(nh * 0.35))
            else:
                trim_axes = (int(nw * 0.35), int(nh * 0.75)) 
        else:
            # Condition B: Fallback to top of shirt
            coords = cv2.findNonZero(shirt_mask)
            if coords is not None:
                x_f, y_f, w_f, h_f = cv2.boundingRect(coords)
                print(f"  ✂ No specific neck label. Fallback trimming at top of shirt: y={y_f}")
                trim_center = (x_f + w_f // 2, y_f + int(h_f * 0.02))
                # USER REQUEST: SHALLOWER FOR COAT
                if is_coat:
                    trim_axes = (int(w_f * 0.15), int(h_f * 0.08))
                else:
                    trim_axes = (int(w_f * 0.26), int(h_f * 0.20))
        
        if trim_center is not None and trim_axes is not None:
            # Draw Half-Oval (180 to 360 degrees for top bite? No, 0-180 is bottom half, 180-360 is top half.
            # actually we want a "bite" from the top. 
            # Ellipse drawing: angle=0, startAngle=0, endAngle=180 draws bottom half. 
            # We want to clear the pixels, so we draw 0.
            # Let's draw a full ellipse to be safe, centered on the top edge. 
            cv2.ellipse(shirt_mask, trim_center, trim_axes, 0, 0, 360, 0, -1)
            print("  ✂ Applied Inner Neck Trim.")
        # ------------------------------------------------
        
        # Save color-coded debug segmentation
        if output_dir:
            h, w = seg_map.shape
            seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_idx in unique_classes:
                if cls_idx in colors:
                    seg_vis[seg_map == cls_idx] = colors[cls_idx]
                elif cls_idx in shirt_classes:
                    seg_vis[seg_map == cls_idx] = [170, 0, 51] # Default shirt red
                else:
                    seg_vis[seg_map == cls_idx] = [128, 128, 128]
            
            # Draw labels
            seg_vis = self.draw_labels(seg_vis, seg_map)
            
            cv2.imwrite(str(output_dir / "0_SEGMENTATION_LEGEND.png"), seg_vis)
            print(f"  ✓ Saved: 0_SEGMENTATION_LEGEND.png (with labels)")
        
        # Save masks
        cv2.imwrite(str(output_dir / "1_SHIRT_MASK.png"), shirt_mask)
        print(f"  ✓ Saved masks")
        
        # Extract shirt
        shirt_data = None
        if np.sum(shirt_mask > 0) > 500:
            # DETECTION: Is it sleeveless?
            # Class 32 is "sleeve". If sleeve pixels < 2% of total shirt pixels, it's sleeveless.
            sleeve_pixels = np.sum(seg_map == 32)
            total_shirt_pixels = np.sum(shirt_mask > 0)
            is_sleeveless = (sleeve_pixels < total_shirt_pixels * 0.02)
            if is_sleeveless:
                print(f"  👗 Detected Sleeveless Garment (Sleeve pixels: {sleeve_pixels})")
            
            coords = cv2.findNonZero(shirt_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                print(f"  Shirt bbox: x={x}, y={y}, w={w}, h={h}")
                
                # Validate bbox
                if x >= 0 and y >= 0 and x + w <= W and y + h <= H:
                    shirt_roi = cloth_img[y:y+h, x:x+w].copy()
                    mask_roi = shirt_mask[y:y+h, x:x+w].copy()
                    
                    shirt_cutout = cv2.bitwise_and(shirt_roi, shirt_roi, mask=mask_roi)
                    
                    # Save with white background
                    white_bg = np.ones_like(shirt_roi) * 255
                    inv_mask = cv2.bitwise_not(mask_roi)
                    white_bg = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
                    shirt_display = cv2.add(shirt_cutout, white_bg)
                    
                    cv2.imwrite(str(output_dir / "2_CUT_SHIRT.png"), shirt_display)
                    print(f"  ✓ Saved: 2_CUT_SHIRT.png")
                    
                    shirt_data = (shirt_cutout, mask_roi, (x, y, w, h), is_sleeveless)
                else:
                    print(f"  ⚠ Invalid shirt bbox!")
        else:
            print(f"  ⚠ No shirt detected")
        
        return shirt_data
    
    def detect_pose(self, image):
        """Detect MediaPipe pose"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return None
        
        H, W = image.shape[:2]
        lms = results.pose_landmarks.landmark
        
        # --- Strict Check for Flat Lay (Ghost Pose) ---
        nose_vis = lms[0].visibility
        le_vis = lms[2].visibility
        re_vis = lms[5].visibility
        
        # 1. Visibility Check
        if nose_vis < 0.9 or le_vis < 0.9 or re_vis < 0.9:
            print(f"  [DEBUG] Low head visibility (N={nose_vis:.2f}, LE={le_vis:.2f}, RE={re_vis:.2f}). Rejecting pose.")
            return None
            
        # 2. Geometric Check (Face printed on shirt?)
        nose_y = lms[0].y
        shoulder_y = (lms[11].y + lms[12].y) / 2
        print(f"  [DEBUG] Nose Y={nose_y:.3f}, Shoulder Y={shoulder_y:.3f}")
        
        if nose_y > shoulder_y:
            print(f"  [DEBUG] Nose below shoulders. Rejecting pose.")
            return None
            
        def to_px(idx):
            return np.array([lms[idx].x * W, lms[idx].y * H], dtype=np.float32)
        
        kp = {
            'left_shoulder': to_px(11),
            'right_shoulder': to_px(12),
            'left_elbow': to_px(13),
            'right_elbow': to_px(14),
            'left_wrist': to_px(15),
            'right_wrist': to_px(16),
            'left_hip': to_px(23),
            'right_hip': to_px(24),
            'left_knee': to_px(25),
            'right_knee': to_px(26),
            'left_ankle': to_px(27),
            'right_ankle': to_px(28),
            # Face landmarks for trimming
            'face_pts': [to_px(i) for i in range(11)]
        }
        
        # Calculate Spine Anchors for Rigid Warping
        ls = kp['left_shoulder']
        rs = kp['right_shoulder']
        lh = kp['left_hip']
        rh = kp['right_hip']
        
        mid_s = (ls + rs) / 2.0
        mid_h = (lh + rh) / 2.0
        spine_vec = mid_h - mid_s
        
        kp['neck_anchor'] = mid_s - (spine_vec * 0.1) # Slightly above shoulders
        kp['torso_mid'] = (mid_s + mid_h) / 2.0
        kp['bottom_center'] = mid_h + (spine_vec * 0.2) # Below hip level
        
        return kp
    
    def get_seg_bbox(self, seg_map, classes):
        """Helper to find bounding box of specific classes"""
        mask = np.isin(seg_map, classes).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            return cv2.boundingRect(coords)
        return None

    def estimate_pose_from_bbox(self, bbox, seg_map, is_sleeveless=False, target_has_hood=False, is_coat=False):
        """
        Estimate keypoints from bounding box (flat lay).
        If HOOD detected, applies an 8% downward offset to source points 
        to effectively move the ENTIRE garment UP on the person.
        """
        x, y, w, h = bbox
        kp = {}
        
        # Check specifically for Hood (28)
        hood_bbox = self.get_seg_bbox(seg_map, [28])
        has_hood = (hood_bbox is not None) or target_has_hood
        
        # PROPORTION TUNING
        if is_coat:
            sy = int(y + h * 0.12) # Coats sit slightly higher on shoulders
            # Narrower shoulder points (0.2-0.8) for coats to increase scale
            lx = int(x + w * 0.84) 
            rx = int(x + w * 0.12)
        else:
            sy = int(y + h * 0.10) if is_sleeveless else int(y + h * 0.15)
            lx = int(x + w * 0.95) if is_sleeveless else int(x + w * 0.8)
            rx = int(x + w * 0.05) if is_sleeveless else int(x + w * 0.2)
        hy = int(y + h * 0.9)
        
        kp['left_shoulder'] = (lx, sy)
        kp['right_shoulder'] = (rx, sy)
        kp['left_hip'] = (lx, hy)
        kp['right_hip'] = (rx, hy)

        # Neck Anchor (Highest point of garment)
        kp['neck_anchor'] = (x + w // 2, y) # Top center of bbox as fallback
        
        # INTERNAL SPINE POINTS for natural flow
        kp['torso_mid'] = (x + w // 2, y + int(h * 0.45))
        kp['bottom_center'] = (x + w // 2, y + h)
            
        return kp

    def warp_clothing_similarity(self, cutout, mask, cloth_kp, person_kp, person_shape, bbox, clothing_type):
        """
        [Legacy Fallback] Warp using Optimal Similarity Transform.
        """
        h, w = cutout.shape[:2]
        x, y, w_box, h_box = bbox
        
        src_points = []
        dst_points = []
        
        kp_names = ['neck_anchor', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if clothing_type == 'shirt':
            kp_names += ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
            
        for name in kp_names:
            if name in cloth_kp and name in person_kp:
                c_pt = np.array(cloth_kp[name])
                c_pt[0] -= x
                c_pt[1] -= y
                src_points.append(c_pt)
                p_pt = np.array(person_kp[name])
                dst_points.append(p_pt)
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        if len(src_points) < 2:
            return None, None
            
        M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
        if M is None: return None, None
        
        ph, pw = person_shape[:2]
        warped = cv2.warpAffine(cutout, M, (pw, ph), flags=cv2.INTER_LANCZOS4)
        warped_mask = cv2.warpAffine(mask, M, (pw, ph), flags=cv2.INTER_NEAREST)
        return warped, warped_mask

    def warp_clothing_tps(self, cutout, mask, cloth_kp, person_kp, person_shape, bbox):
        """
        Warp using Thin Plate Spline (TPS) for non-rigid deformation.
        This allows the garment to 'flow' with the torso.
        """
        h_src, w_src = cutout.shape[:2]
        x_off, y_off, _, _ = bbox
        
        src_pts = []
        dst_pts = []
        
        kp_names = [
            'neck_anchor', 'torso_mid', 'bottom_center',
            'left_shoulder', 'right_shoulder', 
            'left_hip', 'right_hip',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ]
        
        for name in kp_names:
            if name in cloth_kp and name in person_kp:
                # Source Point (Relative to cutout)
                c_pt = np.array(cloth_kp[name], dtype=np.float32)
                c_pt[0] -= x_off
                c_pt[1] -= y_off
                
                # Target Point
                p_pt = np.array(person_kp[name], dtype=np.float32)
                
                src_pts.append(c_pt)
                dst_pts.append(p_pt)
        
        if len(src_pts) < 5:
            print(f"  ⚠ Not enough points for TPS (Need 5+, Got {len(src_pts)}). Falling back to Similarity.")
            return self.warp_clothing_similarity(cutout, mask, cloth_kp, person_kp, person_shape, bbox, 'shirt')
            
        # TPS Transformer
        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # Format for TPS: (1, N, 2)
        # Note: We map DST (Person) -> SRC (Cloth) for the backward warp used by remap
        src_pts = np.array(src_pts).reshape(1, -1, 2)
        dst_pts = np.array(dst_pts).reshape(1, -1, 2)
        
        # Matches are (1, N) from 0 to N-1
        matches = [cv2.DMatch(i, i, 0) for i in range(src_pts.shape[1])]
        
        # Estimate transformation from Target to Source
        tps.estimateTransformation(dst_pts, src_pts, matches) 
        
        # Use remap for robustness (OpenCV 4.11+ warpImage API is picky)
        ph, pw = person_shape[:2]
        grid_y, grid_x = np.mgrid[0:ph, 0:pw].astype(np.float32)
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(1, -1, 2)
        
        # Apply the transformation to every pixel in the target grid
        _, transformed_grid = tps.applyTransformation(grid_pts)
        
        map_x = transformed_grid[0, :, 0].reshape(ph, pw).astype(np.float32)
        map_y = transformed_grid[0, :, 1].reshape(ph, pw).astype(np.float32)
        
        # Perform the Warp
        warped = cv2.remap(cutout, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped, warped_mask



    def process(self, person_img, cloth_img, output_dir, original_img=None, clean_img=None, combined_outfit=False):
        """Main pipeline using B3 for accurate detection"""
        print("="*70)
        print("B3-Integrated Shirt Warping")
        print("="*70)
        
        # Decide which image to use for Segmentation/Pose (Analysis)
        # If original is passed (from helper), use it to detect body parts correctly.
        # If not, use the input person_img (which is assumed to be original in standalone mode).
        analysis_img = original_img if original_img is not None else person_img
        
        # 1. Segment Cloth (B3 for variety)
        print("\nSegmenting cloth (B3)...")
        seg_map_c = self.segment_cloth_b3(cloth_img)
        
        # --- GLOBAL RULE: REMOVE LOWER GARMENTS FROM CLOTH IMAGE ---
        # IDs for lower garments: 7 (pants), 8 (shorts), 9 (skirt)
        lower_garment_mask = np.isin(seg_map_c, [7, 8, 9]).astype(np.uint8) * 255
        if np.sum(lower_garment_mask > 0) > 0:
            print(f"  👖 Detected lower garment in cloth image. Removing it for further use...")
            # Create a white background for the removed areas
            white_bg = np.ones_like(cloth_img) * 255
            # Mask out the lower garment in the cloth image
            cloth_img = np.where(lower_garment_mask[:, :, None] == 255, white_bg, cloth_img)
            
            # Re-segment the cleaned cloth image to ensure seg_map_c matches the new cloth_img
            print("  Re-segmenting cleaned cloth image...")
            seg_map_c = self.segment_cloth_b3(cloth_img)
        # ----------------------------------------------------------
        
        # Segment Person (using Analysis Image)
        # We use B3 for Target Shirt detection
        print("\nSegmenting person (B3)...")
        seg_map_p_b3 = self.segment_cloth_b3(analysis_img)
        
        # We also use B2 for legacy Face/Hair masking on person
        print("Segmenting person (B2 Fallback for Face/Hair masking)...")
        seg_map_p_b2 = self.segment_cloth(analysis_img)
        
        # Save person debug (B3 based)
        colors_b3 = {
            1: [170, 0, 51], 2: [170, 0, 51], 32: [255, 0, 0]
        }
        h, w = seg_map_p_b3.shape
        seg_vis_p = np.zeros((h, w, 3), dtype=np.uint8)
        for u in np.unique(seg_map_p_b3):
            if u in colors_b3: seg_vis_p[seg_map_p_b3 == u] = colors_b3[u]
            else: seg_vis_p[seg_map_p_b3 == u] = [128, 128, 128]
        seg_vis_p = self.draw_labels(seg_vis_p, seg_map_p_b3)
        cv2.imwrite(str(output_dir / "0_PERSON_SEGMENTATION.png"), seg_vis_p)
        print(f"✓ Saved: 0_PERSON_SEGMENTATION.png (B3 labels)")

        # --- CLOTH REMOVAL (Integrated) ---
        try:
            from modules.virtual_tryon2.cloth_remover import ClothRemover
            # Standard Path (from models_downloader.py)
            remover_model_path = self.base_dir / "models" / "SegFormerB2Clothes" / "segformer_b2_clothes.onnx"
            # Fallback Path (Legacy/Local)
            if not remover_model_path.exists():
                remover_model_path = self.base_dir / "models" / "segformer_b2_clothes.onnx"
            
            print(f"Initializing Cloth Remover (shirt mode) from: {remover_model_path}")
            remover = ClothRemover(str(remover_model_path))
            print("Removing original shirt...")
            person_clean, _ = remover.remove_shirt(person_img)
        except Exception as e:
            print(f"Warning: Cloth Removal Failed: {e}. Using original person image.")
            person_clean = person_img.copy()

        # Find Target BBox on Person (B3 Shirt classes: 1, 2, 3, 4, 5, 6, 10)
        target_shirt_mask = np.isin(seg_map_p_b3, [1, 2, 3, 4, 5, 6, 10]).astype(np.uint8)
        target_shirt_bbox = cv2.boundingRect(target_shirt_mask) if np.sum(target_shirt_mask)>0 else None
        
        if target_shirt_bbox: print(f"  Target Shirt found: {target_shirt_bbox}")
        else: print("  ⚠ No Target Shirt found on person")

        # Extract Source Shirt (B3 labels)
        shirt_data = self.extract_and_save_clothing(cloth_img, seg_map_c, output_dir)

        if not shirt_data:
            print("✗ No shirt detected on source!")
            return None

        # 2. Detect Poses
        print("\n[Step 2] Detecting poses...")
        cloth_kp = self.detect_pose(cloth_img)
        # Use Analysis Image (Original) for pose detection
        person_kp = self.detect_pose(analysis_img)
        
        if not cloth_kp:
            print("  ⚠ No cloth pose (Flat Lay). Estimating from Segmentation BBox...")
            # Check if person has hood (Class 28)
            target_has_hood = np.any(seg_map_p_b3 == 28)
            
            shirt_cutout, shirt_mask, s_bbox, is_sleeveless = shirt_data
            # Check if source is a coat for better posing
            is_coat = np.any(seg_map_c == 10)
            cloth_kp = self.estimate_pose_from_bbox(s_bbox, seg_map_c, 
                                                    is_sleeveless=is_sleeveless, 
                                                    target_has_hood=target_has_hood,
                                                    is_coat=is_coat)
            print("    ✓ Estimated Shirt Pose")
        
        if not person_kp:
            print("  ✗ No person pose!")
            return None
        print("  ✓ Person pose detected")

        # 3. Warp Shirt (TPS Non-Rigid Warp)
        print("\n[Step 3] Warping clothing (TPS Non-Rigid Flow)...")
        result = person_clean.copy()
        
        if shirt_data and 'left_shoulder' in person_kp and 'right_shoulder' in person_kp:
            print("  Warping shirt (TPS method)...")
            shirt_cutout, shirt_mask, shirt_bbox, is_sleeveless = shirt_data
            
            # --- DETECTION: Garment Type & Hood ---
            # Check both garment and person for Hood (28)
            garment_has_hood = np.any(seg_map_c == 28)
            garment_is_coat = np.any(seg_map_c == 10) 
            person_has_hood = np.any(seg_map_p_b3 == 28)
            has_hood = garment_has_hood or person_has_hood
            
            if garment_is_coat:
                print("  🧥 Detected Coat (Class 10). Switching to Shoulder-Based scaling.")

            # --- NON-COMPRESSIVE AXIS-RELATIVE PROJECTION (Enhanced) ---
            # 1. Define Person's Spine Axis (Neck -> Hips Mid)
            l_s_p = np.array(person_kp['left_shoulder'], dtype=np.float32)
            r_s_p = np.array(person_kp['right_shoulder'], dtype=np.float32)
            mid_s_p = (l_s_p + r_s_p) / 2.0
            
            # Adaptive Neck Lift: Collars of tanks should be lower than hoodies
            # Coats need a moderate lift to cover shoulders but not too much to lose length
            if garment_is_coat:
                lift_mult = 0.32
            else:
                lift_mult = 0.15 if is_sleeveless else 0.32
                
            h_dist_p = np.linalg.norm(l_s_p - r_s_p)
            
            # --- SEGMENTATION SANITY CHECK FOR WIDTH ---
            # Use segmentation to find the actual width of the person's frame at shoulder level
            # This is robust against arm-crossing poses where landmarks pull inward
            sy_px = int(mid_s_p[1])
            person_width_seg = 0
            if 0 <= sy_px < seg_map_p_b3.shape[0]:
                row = seg_map_p_b3[sy_px, :]
                body_indices = np.where(np.isin(row, [1, 2, 3, 4, 5, 6, 10, 20, 32]))[0]
                if len(body_indices) > 0:
                    person_width_seg = np.max(body_indices) - np.min(body_indices)
            
            p_shoulder_width = max(h_dist_p, person_width_seg)
            if person_width_seg > h_dist_p:
                print(f"  [DEBUG] Shoulder width boosted by segmentation: {h_dist_p:.0f} -> {person_width_seg}")
            
            if 'left_hip' in person_kp and 'right_hip' in person_kp:
                l_h_p = np.array(person_kp['left_hip'], dtype=np.float32)
                r_h_p = np.array(person_kp['right_hip'], dtype=np.float32)
                mid_h_p = (l_h_p + r_h_p) / 2.0
                p_torso_len = np.linalg.norm(mid_h_p - mid_s_p)
                
                # Spine Vector (Tilt of the person)
                spine_vec = mid_h_p - mid_s_p
                spine_unit = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)
                
                # Right Vector (Perpendicular to spine)
                right_unit = np.array([spine_unit[1], -spine_unit[0]])
                
                # RECALCULATE Neck with LEAN and HOOD SHIFT (8% of Torso)
                # Shifting target neck move the ENTIRE garment UP
                hood_shift_px = 0
                if has_hood:
                    hood_shift_px = p_torso_len * 0.10
                    print(f"  ↔ Hood detected: Shifting target UP {hood_shift_px:.1f}px (10%)")

                # --- USER REQUEST: SLEEVELESS LIFT (8% of Torso) ---
                sleeveless_lift_px = 0
                if is_sleeveless:
                    sleeveless_lift_px = p_torso_len * 0.08
                    print(f"  ↥ Sleeveless detected: Lifting target UP {sleeveless_lift_px:.1f}px (8%)")
                
                target_neck = mid_s_p - (spine_unit * (h_dist_p * lift_mult + hood_shift_px + sleeveless_lift_px))
                person_kp['neck_anchor'] = (int(target_neck[0]), int(target_neck[1]))
                
                # 2. Define Source "Spine"
                sx, sy, sw, sh = shirt_bbox
                s_neck = np.array([sx + sw/2.0, sy], dtype=np.float32)
                s_mid_s = np.array([sx + sw/2.0, sy + sh * 0.15], dtype=np.float32) 
                s_mid_h = np.array([sx + sw/2.0, sy + sh * 0.85], dtype=np.float32)
                s_torso_len = np.linalg.norm(s_mid_h - s_mid_s)
                
                # Unified Scale (Preserve Aspect Ratio)
                if garment_is_coat:
                    # Calculate Shoulder-Based Scale
                    # Garment shoulders
                    l_s_g = np.array(cloth_kp['left_shoulder'], dtype=np.float32)
                    r_s_g = np.array(cloth_kp['right_shoulder'], dtype=np.float32)
                    g_shoulder_width = np.linalg.norm(l_s_g - r_s_g)
                    
                    # Person shoulders (from landmark or segmentation)
                    # p_shoulder_width already calculated above
                    
                    # --- POSE RESILIENCE ---
                    # Ensure a minimum width based on torso height to prevent "thin coat" effect.
                    min_width = p_torso_len * 0.85 
                    p_shoulder_width = max(p_shoulder_width, min_width)
                    
                    # Scale based on shoulder width + significant outerwear boost (35% for coats)
                    scale = (p_shoulder_width * 1.35) / (g_shoulder_width + 1e-6)
                    print(f"  ↔ Coat Width Scaling: {scale:.2f} (Target Width: {int(p_shoulder_width)}, Garment: {int(g_shoulder_width)})")
                
                elif is_sleeveless:
                    # --- USER REQUEST: SLEEVELESS STRAP ALIGNMENT (SHOULDER-TO-SHOULDER + BOOST) ---
                    # Measure width of top 15% of garment (stras), IGNORING COLLAR/HOOD.
                    # Target: 125% of Person Shoulder Width to guarantee coverage.
                    
                    # 1. Get ShoulderOnly Mask (Exclude Collar/Hood)
                    sx, sy, sw, sh = shirt_bbox
                    # seg_map_c is the full cloth segmentation (passed to extract_and_save_clothing but we need it here)
                    # We are inside 'main', so 'seg_map_c' is available.
                    
                    if seg_map_c is not None:
                        seg_roi = seg_map_c[sy:sy+sh, sx:sx+sw]
                        # Exclude Collar (29) and Hood (28) to measure actual shoulder straps
                        shoulder_mask_roi = (seg_roi != 29) & (seg_roi != 28) & (shirt_data[1] > 0)
                        shoulder_mask_roi = shoulder_mask_roi.astype(np.uint8) * 255
                    else:
                        shoulder_mask_roi = shirt_data[1]

                    # 2. Measure Top Width
                    mh, mw = shoulder_mask_roi.shape
                    top_slice = shoulder_mask_roi[:int(mh * 0.15), :] 
                    pts = cv2.findNonZero(top_slice)
                    
                    if pts is not None:
                        g_min_x = np.min(pts[:, 0, 0])
                        g_max_x = np.max(pts[:, 0, 0])
                        g_top_width = g_max_x - g_min_x
                        
                        # 3. Target Width (Boosted)
                        # User Request: "technique.. size will be increase" -> 1.25x Factor
                        target_width_p = p_shoulder_width * 1.25
                        
                        scale = target_width_p / (g_top_width + 1e-6)
                        print(f"  🎽 Sleeveless Scaling (No Collar): {scale:.2f} (Strap W: {g_top_width} -> Target: {target_width_p:.1f})")
                    else:
                        print("  ⚠ Sleeveless but no shoulder pixels found? Fallback to torso len.")
                        scale = p_torso_len / (s_torso_len + 1e-6)

                else:
                    scale = p_torso_len / (s_torso_len + 1e-6)
                
                # --- COMBINED OUTFIT CASE: STRICT SHOULDER-EDGE ALIGNMENT ---
                # When the cloth image contains both upper and lower garments (like shirt+pants or blouse+skirt),
                # the upper garment tends to be oversized because it's scaled based on torso instead of shoulders.
                # In this case, we enforce that the top corners of the upper garment align exactly to person's shoulder edges.
                if combined_outfit:
                    print("  🔧 COMBINED OUTFIT MODE: Enforcing strict shoulder-edge alignment...")
                    
                    # Measure the ACTUAL width of the upper garment from the cloth image
                    # (Use the top 25% of the cutout to get shoulder/top width)
                    sx, sy, sw, sh = shirt_bbox
                    cutout_mask = shirt_data[1]  # The extracted mask
                    mh, mw = cutout_mask.shape[:2]
                    top_slice = cutout_mask[:int(mh * 0.25), :]
                    pts = cv2.findNonZero(top_slice)
                    
                    if pts is not None and len(pts) > 10:
                        g_min_x = np.min(pts[:, 0, 0])
                        g_max_x = np.max(pts[:, 0, 0])
                        garment_top_width = g_max_x - g_min_x
                        
                        # Target: person's shoulder width (edge to edge)
                        # p_shoulder_width was already calculated above as max(landmark, segmentation)
                        target_width = p_shoulder_width * 1.0  # Exact 1:1 match, no boost
                        
                        shoulder_scale = target_width / (garment_top_width + 1e-6)
                        
                        # Only apply if the new scale is SMALLER (we want to shrink oversized garments)
                        if shoulder_scale < scale:
                            print(f"  ↔ Garment top width: {garment_top_width:.0f}px, Person shoulders: {p_shoulder_width:.0f}px")
                            print(f"  📏 Scale reduced: {scale:.3f} -> {shoulder_scale:.3f} (enforcing shoulder alignment)")
                            scale = shoulder_scale
                        else:
                            print(f"  ✓ Current scale ({scale:.3f}) already fits within shoulders, no adjustment needed.")
                    else:
                        print("  ⚠ Could not measure garment top width, using default scale.")
                    
                # Increase scale cap to 8.0 for high-res images
                scale = max(0.1, min(scale, 8.0)) 

                # Re-calculate sw, sh based on scale for dots (Actually sw/sh don't change, but projection does)
                
                # --- HIP-LEVEL LENGTH CONSTRAINT (User Request) ---
                # Ensure tops/shirts don't extend below the hips unless they are coats.
                if not garment_is_coat and 'left_hip' in person_kp and 'right_hip' in person_kp:
                     # Person Hip Level (Max Y of hips to be safe/lowest point)
                     l_h_y = person_kp['left_hip'][1]
                     r_h_y = person_kp['right_hip'][1]
                     max_hip_y = max(l_h_y, r_h_y)
                     
                     # Buffer: Allow hanging slightly below hips (User Request: "little bit below")
                     # Add 10% of torso length as a buffer (User Request: "should not exceed 10% below hips")
                     hip_buffer = p_torso_len * 0.10
                     allowed_bottom_y = max_hip_y + hip_buffer

                     # Estimated Garment Bottom using current scale
                     est_bottom_y = target_neck[1] + (sh * scale)
                     
                     if est_bottom_y > allowed_bottom_y:
                          print(f"  🔻 Length Constraint: Garment projected bottom ({est_bottom_y:.1f}) exceeds Hip Level + Buffer ({allowed_bottom_y:.1f}).")
                          
                          # Calculate Max Scale allowed
                          allowed_len = allowed_bottom_y - target_neck[1]
                          new_scale = allowed_len / (sh + 1e-6)
                          
                          print(f"  📏 Resizing S: {scale:.3f} -> {new_scale:.3f} to fit slightly below hips.")
                          scale = new_scale
                
                # 3. Project Landmarks
                # CORE TORSO (Width-Preserving)
                torso_kps = [
                    'left_shoulder', 'right_shoulder', 
                    'left_hip', 'right_hip',
                    'torso_mid', 'bottom_center'
                ]
                # LIMBS (Blended Follow)
                blend_kps = ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
                
                projected = {}
                for name in torso_kps:
                    if name in cloth_kp:
                        # Project onto Person's Spine Axis (Maintain width)
                        lat_off = cloth_kp[name][0] - (sx + sw / 2.0)
                        long_off = cloth_kp[name][1] - sy # from source neck
                        p_pos = target_neck + (spine_unit * long_off * scale) + (right_unit * lat_off * scale)
                        projected[name] = (int(p_pos[0]), int(p_pos[1]))
                
                for name in blend_kps:
                    if name in cloth_kp and name in person_kp:
                        # Rigid garment pos
                        lat_off = cloth_kp[name][0] - (sx + sw/2.0)
                        long_off = cloth_kp[name][1] - sy
                        rigid_pos = target_neck + (spine_unit * long_off * scale) + (right_unit * lat_off * scale)
                        
                        # Person joint pos
                        person_joint = np.array(person_kp[name], dtype=np.float32)
                        
                        # Blend factor (0.5 for natural flow)
                        blend_factor = 0.5
                        final_pos = (rigid_pos * blend_factor) + (person_joint * (1.0 - blend_factor))
                        projected[name] = (int(final_pos[0]), int(final_pos[1]))
                
                # --- STABILIZE BOTTOM CENTER FOR RIGID WARP ---
                # This prevents the shirt from "melting" or shearing at the base.
                if 'bottom_center' in cloth_kp:
                    bc_off = cloth_kp['bottom_center'][1] - sy
                    bc_p = target_neck + (spine_unit * bc_off * scale)
                    projected['bottom_center'] = (int(bc_p[0]), int(bc_p[1]))

                # Update person_kp
                person_kp.update(projected)
                print(f"  ✓ Refined Projection (Scale={scale:.2f}, Lift={lift_mult:.2f})")

            # Use Rigid 3-Point Affine Warp (Skirt Style)
            warped_shirt, warped_mask = self.warp_clothing_rigid(
                shirt_cutout, shirt_mask, cloth_kp, person_kp,
                person_img.shape, shirt_bbox
            )
            
            if warped_shirt is not None:
                # --- Precision Face Trimming with B2 + MediaPipe Fallback ---
                # User specified: Trim ONLY the face to avoid cutting into the garment
                # 1. B2 Segmentation for Face
                exclusion_mask = np.isin(seg_map_p_b2, [13]).astype(np.uint8) * 255
                
                # 2. MediaPipe Landmark Fallback (Robust against overlap)
                if 'face_pts' in person_kp:
                    pts = np.array(person_kp['face_pts'], dtype=np.int32)
                    # Safety Filter: Only use face landmarks in the upper 40% of the image 
                    # to prevent "ghost" landmarks from creating huge hull trims over the legs.
                    upper_target = int(person_img.shape[0] * 0.4)
                    valid_pts = pts[pts[:, 1] < upper_target]
                    
                    if len(valid_pts) > 2:
                        hull = cv2.convexHull(valid_pts)
                        cv2.fillPoly(exclusion_mask, [hull], 255)
                    
                    # Also add a circle for the mouth area specifically if hoodie is high
                    # Points 9, 10 are mouth corners
                    m_left, m_right = pts[9], pts[10]
                    m_center = ((m_left[0] + m_right[0]) // 2, (m_left[1] + m_right[1]) // 2)
                    # Radius based on mouth width
                    m_radius = int(np.linalg.norm(np.array(m_left) - np.array(m_right)) * 1.5)
                    cv2.circle(exclusion_mask, m_center, m_radius, 255, -1)
                
                # Use a slightly larger dilation for a safer margin around the face (5x5)
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                exclusion_mask = cv2.dilate(exclusion_mask, kernel_dilate, iterations=1)
                
                # --- LEG PROTECTION GUARD ---
                # Force the exclusion mask to be empty below the chest area.
                # This prevents any part of the coat from being trimmed on the legs.
                if 'left_shoulder' in person_kp and 'right_shoulder' in person_kp:
                    shoulder_y_avg = (person_kp['left_shoulder'][1] + person_kp['right_shoulder'][1]) // 2
                    # Everything 15% of torso length below shoulders is protected
                    v_limit = int(shoulder_y_avg + p_torso_len * 0.15)
                    exclusion_mask[v_limit:, :] = 0
                
                # Trim the warped shirt from the protected areas
                warped_mask = cv2.bitwise_and(warped_mask, cv2.bitwise_not(exclusion_mask))
                
                # --- [DISABLED per user request] ---
                # target_original = original_img if original_img is not None else person_img
                # mp_mask = self.masker.get_hands_only_mask(target_original)
                # print("👐 Trimming cloth from Hands (Precise)...")
                # inv_mp_mask = cv2.bitwise_not(mp_mask)
                # warped_mask = cv2.bitwise_and(warped_mask, inv_mp_mask)
                
                # Smooth blending
                mask_smooth = cv2.GaussianBlur(warped_mask, (7, 7), 0)
                alpha = mask_smooth.astype(float) / 255.0
                alpha = np.stack([alpha] * 3, axis=2)
                result = (warped_shirt * alpha + result * (1 - alpha)).astype(np.uint8)
                print("  ✓ Shirt applied (with B2 + MediaPipe Face trim)")

                # --- UNIFIED MASK GENERATION (USER REQUESTED) ---
                print("🎭 Generating Unified VTON Mask...")
                # Use 'shirt' mode for Shirt Warper
                # Pass accurate Original and Clean images for Diff calculation
                target_original = original_img if original_img is not None else person_img # In standalone, person_img IS original
                target_clean = clean_img if clean_img is not None else person_clean
                
                unified_mask = self.masker.get_final_mask(target_original, warped_mask, mode='shirt', clean_img=target_clean)
                
                # --- ARTIFICIAL SKIN HELPER (Integrated) ---
                print("  🦴 Drawing Artificial Skin Skeleton (Arms Only)...")
                # Use target_original for sampling to avoid sampling the applied shirt
                # PASS NO upper_body_mask for shirt.py to avoid clipping by original clothes
                result, skin_mask = self.skin_helper.process(
                    result, warped_mask, output_dir, 
                    pose_kps=person_kp, include_arms=True, include_legs=False, 
                    sampling_img=target_original, upper_body_mask=None
                )
                
                # USER REQUEST: Expand artificial arm mask ONLY in the final mask
                print("  🎭 Expanding Artificial Skin Mask for FINAL_MASK...")
                k_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
                expanded_skin_mask = cv2.dilate(skin_mask, k_skin, iterations=1)
                
                # Add expanded artificial skin mask to the unified mask
                unified_mask = cv2.bitwise_or(unified_mask, expanded_skin_mask)

                # --- FINAL FACE PROTECTION (Override) ---
                # Ensure no part of the mask covers the face/hair
                print("  🛡️ Applying Final Face Shield...")
                face_shield = self.masker.get_face_shield(target_original)
                unified_mask[face_shield > 0] = 0
                
                # Save into output dir
                mask_out = output_dir / "final_mask.png"
                cv2.imwrite(str(mask_out), unified_mask)
                print(f"✅ Saved Unified Mask to: {mask_out}")
                
        print("\n" + "="*70)
        print("✓ Complete!")
        print("="*70)
        
        return result

    def warp_clothing_rigid(self, cutout, mask, cloth_kp, person_kp, person_shape, bbox):
        """
        Rigid 3-Point Affine Warp (Skirt Style from pants.py).
        Maps source shoulders and bottom center to target positions using a simple affine transform.
        This provides cleaner, more linear flow than TPS for structured garments.
        """
        h_src, w_src = cutout.shape[:2]
        x_off, y_off, _, _ = bbox
        
        # Define the 3 anchor points for the transform
        # We need: left_shoulder, right_shoulder, bottom_center
        required_kps = ['left_shoulder', 'right_shoulder', 'bottom_center']
        
        # Check if we have all required keypoints
        for kp_name in required_kps:
            if kp_name not in cloth_kp or kp_name not in person_kp:
                print(f"  ⚠ Missing keypoint '{kp_name}' for Rigid Warp. Falling back to TPS.")
                return self.warp_clothing_tps(cutout, mask, cloth_kp, person_kp, person_shape, bbox)
        
        # Source points (relative to cutout)
        src_pts = []
        for kp_name in required_kps:
            c_pt = np.array(cloth_kp[kp_name], dtype=np.float32)
            c_pt[0] -= x_off
            c_pt[1] -= y_off
            src_pts.append(c_pt)
        
        # Target points (on person)
        dst_pts = []
        for kp_name in required_kps:
            p_pt = np.array(person_kp[kp_name], dtype=np.float32)
            dst_pts.append(p_pt)
        
        # Convert to numpy arrays
        src_tri = np.float32(src_pts)
        dst_tri = np.float32(dst_pts)
        
        # Compute affine transform
        M = cv2.getAffineTransform(src_tri, dst_tri)
        
        # Apply transform
        ph, pw = person_shape[:2]
        warped = cv2.warpAffine(cutout, M, (pw, ph), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_mask = cv2.warpAffine(mask, M, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        print("  ✓ Applied Rigid 3-Point Affine Warp (Skirt Style)")
        return warped, warped_mask


def main():
    # Discover project root (2 levels up from Modules/Virtual_Tryon2 finds Fooocus root)
    base_dir = Path(__file__).resolve().parents[2]
    
    # Standardized paths from models_downloader.py
    B2_MODEL = base_dir / "models" / "SegFormerB2Clothes" / "segformer_b2_clothes.onnx"
    # Fallback for legacy
    if not B2_MODEL.exists():
        B2_MODEL = base_dir / "models" / "segformer_b2_clothes.onnx"
        
    B3_MODEL = base_dir / "models" / "segformer-b3-fashion"
    
    # Check if models exist
    if not B2_MODEL.exists():
        print(f"❌ Error: B2 model not found at {B2_MODEL}")
        return
    if not B3_MODEL.exists():
        print(f"❌ Error: B3 model not found at {B3_MODEL}")
        return
        
    warper = FixedShirtPantsWarper(B2_MODEL, B3_MODEL)
    
    # Example paths
    PERSON_PATH = r"person.png"
    CLOTH_PATH = r"cloth.png"
    OUTPUT_DIR = Path("./output_test")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    person = cv2.imread(PERSON_PATH)
    cloth = cv2.imread(CLOTH_PATH)
    
    if person is None or cloth is None:
        print("✗ Failed to load images!")
        return
    
    result = warper.process(person, cloth, OUTPUT_DIR)
    
    if result is not None:
        cv2.imwrite(str(OUTPUT_DIR / "final_result.png"), result)
        print(f"\n✓ Saved: final_result.png")


if __name__ == "__main__":
    main()