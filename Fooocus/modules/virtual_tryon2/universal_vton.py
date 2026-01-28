try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

import cv2
import numpy as np
import onnxruntime as ort
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path
from modules.virtual_tryon2.vton_masking_helper import VTONMasker
from modules.virtual_tryon2.artificial_skin_helper import ArtificialSkinHelper

class UniversalGarmentWarper:
    def __init__(self, b2_model_path, b3_model_path):
        # B2 ONNX
        print(f"Loading SegFormer-B2 ONNX from: {b2_model_path}")
        self.b2_session = ort.InferenceSession(str(b2_model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.b2_input_name = self.b2_session.get_inputs()[0].name
        
        # B3 Torch
        print(f"Loading SegFormer-B3 from: {b3_model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.b3_processor = SegformerImageProcessor.from_pretrained(b3_model_path)
        self.b3_model = SegformerForSemanticSegmentation.from_pretrained(b3_model_path).to(self.device).eval()
        
        # MediaPipe Pose with Fallback
        self.pose = None
        if HAS_MEDIAPIPE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
                print("✓ MediaPipe Pose initialized")
            except Exception as e:
                print(f"⚠ MediaPipe initialization failed: {e}. Using segmentation-based pose estimation.")
        else:
            print("⚠ MediaPipe not available. Using segmentation-based pose estimation.")
            
        # Discover project root (4 levels up from Modules/Virtual_Tryon2)
        self.base_dir = Path(__file__).resolve().parents[3]
        
        # Unified Masking Helper
        sam_path = self.base_dir / "models" / "sam" / "sam_vit_b_01ec64.pth"
        self.masker = VTONMasker(
            seg_model_path=str(b2_model_path), 
            b3_model_path=str(b3_model_path),
            sam_model_path=str(sam_path) if sam_path.exists() else None
        )
        
        # Artificial Skin Helper
        self.skin_helper = ArtificialSkinHelper()

    # =========================================================================
    # SEGMENTATION & POSE
    # =========================================================================

    def segment_b3(self, image):
        """Standard SegFormer-B3 Fashion Segmentation"""
        H, W = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.b3_processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.b3_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def segment_b2(self, image):
        """Auxiliary B2 Segmentation for skin/hair (ONNX)"""
        H, W = image.shape[:2]
        img_resized = cv2.resize(image, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_norm = (img_rgb / 255.0 - mean) / std
        img_tensor = img_norm.transpose(2, 0, 1).astype(np.float32)[None, ...]
        outputs = self.b2_session.run(None, {self.b2_input_name: img_tensor})
        logits = outputs[0][0].transpose(1, 2, 0)
        logits_hr = cv2.resize(logits, (W, H), interpolation=cv2.INTER_LINEAR)
        return np.argmax(logits_hr, axis=2).astype(np.uint8)

    def detect_pose(self, image, b3_seg=None):
        H, W = image.shape[:2]
        if self.pose:
            try:
                res = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    def to_px(idx): return np.array([lms[idx].x * W, lms[idx].y * H], dtype=np.float32)
                    face_vis = [lms[i].visibility for i in range(11)]
                    if np.mean(face_vis) > 0.5:
                        return {
                            'left_shoulder': to_px(11), 'right_shoulder': to_px(12),
                            'left_elbow': to_px(13), 'right_elbow': to_px(14),
                            'left_wrist': to_px(15), 'right_wrist': to_px(16),
                            'left_hip': to_px(23), 'right_hip': to_px(24),
                            'left_knee': to_px(25), 'right_knee': to_px(26),
                            'left_ankle': to_px(27), 'right_ankle': to_px(28),
                            'nose': to_px(0), 'face_pts': [to_px(i) for i in range(11)]
                        }
            except Exception:
                pass
        
        # Fallback: Estimate from Segmentation
        print("  Using segmentation-based pose estimation for person...")
        if b3_seg is None:
            b3_seg = self.segment_b3(image)
        
        return self._estimate_person_pose_from_seg(b3_seg, H, W)

    def _estimate_person_pose_from_seg(self, seg, H, W):
        """Estimates keypoints by analyzing the person's segmentation mask"""
        # Upper body (classes 1-6, 10, 11, 12)
        upper_mask = np.isin(seg, [1, 2, 3, 4, 5, 6, 10, 11, 12]).astype(np.uint8) * 255
        coords = cv2.findNonZero(upper_mask)
        if coords is None: return None
        
        x, y, w, h = cv2.boundingRect(coords)
        
        kp = {
            'left_shoulder': np.array([x + w*0.8, y + h*0.2], dtype=np.float32),
            'right_shoulder': np.array([x + w*0.2, y + h*0.2], dtype=np.float32),
            'left_hip': np.array([x + w*0.75, y + h*0.9], dtype=np.float32),
            'right_hip': np.array([x + w*0.25, y + h*0.9], dtype=np.float32),
            'nose': np.array([x + w//2, y], dtype=np.float32)
        }
        
        # Elbows/Wrists fallback for TPS
        kp['left_elbow'] = kp['left_shoulder'] + [w*0.1, h*0.3]
        kp['right_elbow'] = kp['right_shoulder'] - [w*0.1, -h*0.3]
        kp['left_wrist'] = kp['left_elbow'] + [w*0.05, h*0.2]
        kp['right_wrist'] = kp['right_elbow'] - [w*0.05, -h*0.2]
        
        # Check for legs (classes 7, 8, 9)
        lower_mask = np.isin(seg, [7, 8, 9]).astype(np.uint8) * 255
        l_coords = cv2.findNonZero(lower_mask)
        if l_coords is not None:
            lx, ly, lw, lh = cv2.boundingRect(l_coords)
            kp['left_knee'] = np.array([lx + lw*0.75, ly + lh*0.5], dtype=np.float32)
            kp['right_knee'] = np.array([lx + lw*0.25, ly + lh*0.5], dtype=np.float32)
            kp['left_ankle'] = np.array([lx + lw*0.75, ly + lh*0.9], dtype=np.float32)
            kp['right_ankle'] = np.array([lx + lw*0.25, ly + lh*0.9], dtype=np.float32)
        else:
            kp['left_knee'] = kp['left_hip'] + [0, h*0.5]
            kp['right_knee'] = kp['right_hip'] + [0, h*0.5]
            kp['left_ankle'] = kp['left_knee'] + [0, h*0.5]
            kp['right_ankle'] = kp['right_knee'] + [0, h*0.5]
            
        return kp

    # =========================================================================
    # CORE LOGIC: TPS & LBS
    # =========================================================================

    def get_tps_warp(self, cutout, mask, src_kp, dst_kp, target_shape, bbox):
        """TPS warping for flexible garments (Upper/Full/Scarf)"""
        x_off, y_off = bbox[:2]
        kp_names = ['neck_anchor', 'torso_mid', 'bottom_center', 'left_shoulder', 'right_shoulder', 
                    'left_hip', 'right_hip', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        
        s_pts, d_pts = [], []
        for name in kp_names:
            if name in src_kp and name in dst_kp:
                s_pts.append(src_kp[name] - [x_off, y_off])
                d_pts.append(dst_kp[name])
        
        if len(s_pts) < 5: return None, None
        
        tps = cv2.createThinPlateSplineShapeTransformer()
        s_pts_arr = np.array(s_pts).reshape(1, -1, 2)
        d_pts_arr = np.array(d_pts).reshape(1, -1, 2)
        matches = [cv2.DMatch(i, i, 0) for i in range(s_pts_arr.shape[1])]
        tps.estimateTransformation(d_pts_arr, s_pts_arr, matches)
        
        ph, pw = target_shape[:2]
        grid_y, grid_x = np.mgrid[0:ph, 0:pw].astype(np.float32)
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(1, -1, 2)
        _, trans_grid = tps.applyTransformation(grid_pts)
        map_x = trans_grid[0, :, 0].reshape(ph, pw).astype(np.float32)
        map_y = trans_grid[0, :, 1].reshape(ph, pw).astype(np.float32)
        
        warped = cv2.remap(cutout, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped, warped_mask

    def get_lbs_warp(self, pants_img, src_pose, dst_pose, target_shape):
        """Bone-based skinning (LBS) for Lower garments"""
        h, w = target_shape[:2]
        bones_list = ['waist', 'l_thigh', 'l_shin', 'r_thigh', 'r_shin']
        
        # 1. Inverse Transforms (Target -> Source)
        src_bones = self._get_bones(src_pose)
        dst_bones = self._get_bones(dst_pose)
        s_hw = np.linalg.norm(src_pose['left_hip'] - src_pose['right_hip'])
        d_hw = np.linalg.norm(dst_pose['left_hip'] - dst_pose['right_hip'])
        base_scale = s_hw / (d_hw + 1e-6)
        
        transforms = {}
        for name in bones_list:
            if name in src_bones and name in dst_bones:
                transforms[name] = self._compute_aniso_affine(dst_bones[name][0], dst_bones[name][1], 
                                                             src_bones[name][0], src_bones[name][1], base_scale)
        
        # 2. skinning Weights
        weights_map = self._compute_skinning_weights((h, w), dst_pose, bones_list)
        
        # 3. Accumulate deformation
        y, x = np.mgrid[0:h, 0:w]
        P_dst = np.stack([x, y, np.ones_like(x)], axis=0).reshape(3, -1)
        P_src_acc = np.zeros((2, h*w), dtype=np.float32)
        
        for i, b_name in enumerate(bones_list):
            if b_name in transforms:
                P_src_acc += (transforms[b_name] @ P_dst) * weights_map[:, :, i].reshape(-1)
        
        mx, my = P_src_acc[0, :].reshape(h, w), P_src_acc[1, :].reshape(h, w)
        return cv2.remap(pants_img, mx, my, cv2.INTER_LANCZOS4, borderValue=(0,0,0))

    def _get_bones(self, p):
        return {
            'waist': (p['left_hip'], p['right_hip']),
            'l_thigh': (p['left_hip'], p['left_knee']), 'l_shin': (p['left_knee'], p['left_ankle']),
            'r_thigh': (p['right_hip'], p['right_knee']), 'r_shin': (p['right_knee'], p['right_ankle'])
        }

    def _compute_aniso_affine(self, s1, s2, d1, d2, width_scale):
        sv = s2 - s1
        dv = d2 - d1
        slen, dlen = np.linalg.norm(sv), np.linalg.norm(dv)
        if slen < 1e-6 or dlen < 1e-6: return np.eye(2, 3, dtype=np.float32)
        sx, sy = dlen/slen, width_scale
        asrc, adst = np.arctan2(sv[1], sv[0]), np.arctan2(dv[1], dv[0])
        
        T1 = np.array([[1, 0, -s1[0]], [0, 1, -s1[1]], [0, 0, 1]])
        R1 = np.array([[np.cos(-asrc), -np.sin(-asrc), 0], [np.sin(-asrc), np.cos(-asrc), 0], [0, 0, 1]])
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        R2 = np.array([[np.cos(adst), -np.sin(adst), 0], [np.sin(adst), np.cos(adst), 0], [0, 0, 1]])
        T2 = np.array([[1, 0, d1[0]], [0, 1, d1[1]], [0, 0, 1]])
        return (T2 @ R2 @ S @ R1 @ T1)[:2, :].astype(np.float32)

    def _compute_skinning_weights(self, shape, pose, bones_list, sigma=45.0):
        h, w = shape
        scale = 0.5
        sh, sw = int(h*scale), int(w*scale)
        sy, sx = np.mgrid[0:sh, 0:sw]
        pts = np.stack([sx, sy], axis=2).reshape(-1, 2).astype(np.float32) / scale
        coords = self._get_bones(pose)
        
        weights = np.zeros((sh * sw, len(bones_list)), dtype=np.float32)
        for i, b_name in enumerate(bones_list):
            a, b = coords[b_name]
            ab = b - a
            lab2 = np.sum(ab**2)
            if lab2 < 1e-6: 
                d = np.linalg.norm(pts - a, axis=1)
            else:
                t = np.clip(np.sum((pts - a) * ab, axis=1) / lab2, 0, 1)
                d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
            weights[:, i] = -(d**2) / (2 * sigma**2)
        
        # Softmax
        row_max = np.max(weights, axis=1, keepdims=True)
        exp_w = np.exp(weights - row_max)
        weights = exp_w / np.sum(exp_w, axis=1, keepdims=True)
        weights = weights.reshape(sh, sw, len(bones_list))
        
        full_w = [cv2.resize(weights[:,:,i], (w, h)) for i in range(len(bones_list))]
        return np.stack(full_w, axis=2)

    # =========================================================================
    # REFACTORED: UNIFIED GARMENT HANDLING (As requested)
    # =========================================================================

    def process(self, person_img, cloth_img, out_dir, original_img=None, clean_img=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        
        # Determine images
        analysis_img = original_img if original_img is not None else person_img
        
        print("Segmenting person & cloth...")
        # Use Analysis (Original) for segmentation
        p_seg = self.segment_b3(analysis_img)
        c_seg = self.segment_b3(cloth_img)
        p_seg_b2 = self.segment_b2(analysis_img)
        
        # --- CLOTH REMOVAL (Integrated) ---
        if clean_img is not None:
             print("ℹ Universal: Using passed Clean Image.")
             person_clean = clean_img
        else:
            try:
                remover_model_path = self.base_dir / "models" / "segformer_b2_clothes.onnx"
                print(f"Initializing Cloth Remover (universal mode) from: {remover_model_path}")
                remover = ClothRemover(str(remover_model_path))
                print("Removing ALL clothes...")
                person_clean, _ = remover.remove_all(person_img)
            except Exception as e:
                print(f"Warning: Cloth Removal Failed: {e}. Using original person image.")
                person_clean = person_img.copy()

        # 1. Detect Poses (with fallback)
        p_pose = self.detect_pose(analysis_img, p_seg)
        if not p_pose: return print("✗ No person pose detected!")
        
        # 2. Extract UNIFIED Garment (All Classes 1-46 as requested)
        fashion_ids = list(range(1, 47))
        c_mask = np.isin(c_seg, fashion_ids).astype(np.uint8) * 255
        if np.sum(c_mask > 0) < 500: return print("✗ No garment detected on cloth image!")
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # --- STRAPLESS DETECTION (Early) ---
        has_g_sleeves = np.sum(c_seg == 32) > 100
        has_g_collar = np.sum(c_seg == 29) > 100
        has_g_neckline = np.sum(c_seg == 34) > 100
        is_strapless_early = (not has_g_sleeves and not has_g_collar and not has_g_neckline)
        
        if is_strapless_early:
            print("  👗 Early Strapless Detection: Skipping Neck Trim.")
        else:
            # --- USER REQUEST: TRIM INNER NECK/LABEL AREA (Mirrored from shirt.py) ---
            NECK_LABELS = [28, 29, 34] # Hood, Collar, Neckline
            unique_c_classes = np.unique(c_seg)
            is_coat = (10 in unique_c_classes)
            
            neck_mask = np.isin(c_seg, NECK_LABELS).astype(np.uint8) * 255
            trim_center = None
            trim_axes = None
            
            if np.sum(neck_mask > 0) > 0:
                nx, ny, nw, nh = cv2.boundingRect(cv2.findNonZero(neck_mask))
                trim_center = (nx + nw // 2, ny)
                if is_coat:
                    trim_axes = (int(nw * 0.25), int(nh * 0.35))
                else:
                    trim_axes = (int(nw * 0.35), int(nh * 0.75)) 
            else:
                coords = cv2.findNonZero(c_mask)
                if coords is not None:
                    x_f, y_f, w_f, h_f = cv2.boundingRect(coords)
                    trim_center = (x_f + w_f // 2, y_f + int(h_f * 0.02))
                    if is_coat:
                        trim_axes = (int(w_f * 0.15), int(h_f * 0.08))
                    else:
                        trim_axes = (int(w_f * 0.26), int(h_f * 0.20))
            
            if trim_center is not None and trim_axes is not None:
                cv2.ellipse(c_mask, trim_center, trim_axes, 0, 0, 360, 0, -1)
                print(f"  ✂ Applied Universal Neck Trim ({'Coat' if is_coat else 'Standard'}).")
        # -------------------------------------------------------------------------
        
        # RESOLUTION NORMALIZATION: Resize to standard dimensions for consistent scaling
        coords = cv2.findNonZero(c_mask)
        tight_bbox = cv2.boundingRect(coords)
        tx, ty, tw, th = tight_bbox
        
        # Normalize to standard height (1024px) with minimal 5% padding
        standard_height = 1024
        scale_factor = standard_height / th
        
        # Calculate normalized dimensions
        norm_w = int(tw * scale_factor)
        norm_h = standard_height
        
        # Add minimal 5% padding
        pad_percent = 0.05
        pad_w = int(norm_w * pad_percent)
        pad_h = int(norm_h * pad_percent)
        
        final_w = norm_w + 2 * pad_w
        final_h = norm_h + 2 * pad_h
        
        # Create normalized image
        normalized_cloth = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        normalized_mask = np.zeros((final_h, final_w), dtype=np.uint8)
        normalized_seg = np.zeros((final_h, final_w), dtype=np.uint8)
        
        # Extract and resize the tight-cropped garment
        tight_cloth = cloth_img[ty:ty+th, tx:tx+tw]
        tight_mask = c_mask[ty:ty+th, tx:tx+tw]
        tight_seg = c_seg[ty:ty+th, tx:tx+tw]
        
        resized_cloth = cv2.resize(tight_cloth, (norm_w, norm_h), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(tight_mask, (norm_w, norm_h), interpolation=cv2.INTER_NEAREST)
        resized_seg = cv2.resize(tight_seg, (norm_w, norm_h), interpolation=cv2.INTER_NEAREST)
        
        # Place in center with padding
        normalized_cloth[pad_h:pad_h+norm_h, pad_w:pad_w+norm_w] = resized_cloth
        normalized_mask[pad_h:pad_h+norm_h, pad_w:pad_w+norm_w] = resized_mask
        normalized_seg[pad_h:pad_h+norm_h, pad_w:pad_w+norm_w] = resized_seg
        
        # Replace with normalized versions
        cloth_img = normalized_cloth
        c_mask = normalized_mask
        c_seg = normalized_seg
        
        print(f"  📐 Resolution normalized: {tight_bbox} -> {final_h}x{final_w}px (scale: {scale_factor:.2f}x)")
        
        coords = cv2.findNonZero(c_mask)
        c_bbox = cv2.boundingRect(coords)
        bx, by, bw, bh = c_bbox
        
        print(f"  Unified Garment bbox: {c_bbox}")
        
        # 3. Precision Scaling & Alignment (Anti-Stretching: Sleeve-to-Sleeve)
        # Person Arm Span: Distance between wrists (or elbows if wrists not visible)
        l_w_p = p_pose.get('left_wrist', p_pose['left_shoulder'])
        r_w_p = p_pose.get('right_wrist', p_pose['right_shoulder'])
        p_span = np.linalg.norm(l_w_p - r_w_p)
        p_s_mid = (p_pose['left_shoulder'] + p_pose['right_shoulder']) / 2.0
        
        # Garment Source Pose (Mask-Based with Segmentation Classes)
        c_kp = self.estimate_source_pose_mask_based(c_mask, c_bbox, g_anchor_pt=None, c_seg=c_seg)
        g_span = np.linalg.norm(c_kp['left_wrist'] - c_kp['right_wrist'])
        g_anchor = c_kp['neck_anchor']
        
        # TORSO WIDTHS for validation
        p_t_width = np.linalg.norm(p_pose['left_shoulder'] - p_pose['right_shoulder'])
        g_t_width = np.linalg.norm(c_kp['left_shoulder'] - c_kp['right_shoulder'])
        min_shoulder_y = min(p_pose['left_shoulder'][1], p_pose['right_shoulder'][1])

        # SLEEVELESS DETECTION (Case 3)
        # Check if garment has sleeves (class 32)
        has_g_sleeves = False
        if c_seg is not None:
            has_g_sleeves = np.sum(c_seg == 32) > 100
        
        is_case_3 = (not has_g_sleeves)
        if is_case_3:
            print(f"  🎽 Case 3: Sleeveless garment detected - applying strict inner-arm fit.")

        # Calculate Scale: Match total sleeve-to-sleeve width
        scale_width = (p_span / (g_span + 1e-6))
        
        # Check for pants/jeans in cloth (Class 7 or 8)
        has_pants = False
        has_sequins = False
        if c_seg is not None:
            pants_mask = np.isin(c_seg, [7, 8]).astype(np.uint8) * 255
            if np.sum(pants_mask > 0) > 500:
                has_pants = True
                print("  👖 Pants detected in outfit! Calculating height-based scale...")
            
            # Check for Sequins (Class 45)
            sequin_mask = (c_seg == 45).astype(np.uint8) * 255
            if np.sum(sequin_mask > 0) > 500:
                has_sequins = True
                print("  ✨ Sequins detected (class 45)! Setting target length to 5% KEY below feet...")

        scale = scale_width
        
        # If pants/sequins are present, prioritize HEIGHT scaling
        if (has_pants or has_sequins) and 'left_ankle' in p_pose and 'right_ankle' in p_pose:
            # Person height target logic
            p_min_shoulder_y = min(p_pose['left_shoulder'][1], p_pose['right_shoulder'][1])
            p_avg_ankle_y = (p_pose['left_ankle'][1] + p_pose['right_ankle'][1]) / 2.0
            
            if has_sequins:
                # Sequins target: 5% BELOW feet
                p_height_target = (p_avg_ankle_y + (person_img.shape[0] * 0.05)) - p_min_shoulder_y
            else:
                # Pants target: AT feet/ankles
                p_height_target = p_avg_ankle_y - p_min_shoulder_y
            
            # Garment source height (Anchor to Bottom of BBox)
            g_bottom_y = by + bh
            g_height_source = g_bottom_y - g_anchor[1]
            
            scale_height = p_height_target / (g_height_source + 1e-6)
            scale = scale_height
            
            target_desc = "Ankle + 5%" if has_sequins else "Ankle"
            print(f"  📏 Height scaling ({target_desc}): {scale:.2f} (Target: {int(p_height_target)}, Source: {int(g_height_source)})")
        else:
            print(f"  ↔ Anti-Stretching Scaling: {scale:.2f} (Target Span: {int(p_span)}, Garment Span: {int(g_span)})")
        
        # Ensure minimal scale for torso fit if span is too small (e.g. arms at sides)
        p_t_width = np.linalg.norm(p_pose['left_shoulder'] - p_pose['right_shoulder'])
        p_torso_height = np.linalg.norm(p_pose['left_hip'] - p_pose['left_shoulder'])
        g_t_width = np.linalg.norm(c_kp['left_shoulder'] - c_kp['right_shoulder'])
        min_scale = p_t_width / (g_t_width + 1e-6)
        
        if is_strapless_early:
            # Strapless/Tube Gown Logic: Scale based ONLY on Torso Width (Shoulders)
            g_t_width = np.linalg.norm(c_kp['left_shoulder'] - c_kp['right_shoulder'])
            p_t_width = np.linalg.norm(p_pose['left_shoulder'] - p_pose['right_shoulder'])
            scale = p_t_width / (g_t_width + 1e-6)
            print(f"  👗 Strapless scaling (Torso Width): {scale:.2f}")
        elif is_case_3:
            # Case 3: Sleeveless - Prioritize fitting STRICTLY INSIDE the arms
            # Multi-row scan to find the NARROWEST point between person's arms
            p_inner_width = p_t_width 
            if p_seg_b2 is not None:
                gaps = []
                for offset in range(0, 40, 5): # Scan 8 rows in shoulder/arm area
                    scan_y = int(min_shoulder_y + offset)
                    if scan_y >= person_img.shape[0]: break
                    row_b2 = p_seg_b2[scan_y, :]
                    coords_14 = np.where(row_b2 == 14)[0] 
                    coords_15 = np.where(row_b2 == 15)[0] 
                    if len(coords_14) > 0 and len(coords_15) > 0:
                        gaps.append(np.min(coords_14) - np.max(coords_15))
                
                if gaps:
                    p_inner_width = min(gaps)
                    print(f"  📐 Case 3 Min Inner-Arm Gap: {int(p_inner_width)} (from {len(gaps)} rows)")
            
            # Scale so garment OUTER width fits EXACTLY INSIDE the gap (1.0 multiplier)
            g_outer_width = c_kp.get('outer_shoulder_width', (c_kp['left_wrist'][0] - c_kp['right_wrist'][0]))
            # Robust scaling guard: ensure scale is positive and significant
            g_outer_width = max(10, abs(g_outer_width))
            scale = (p_inner_width / (g_outer_width + 1e-6)) * 1.0 
            print(f"  ↔ Case 3 Exact Inner scaling: {scale:.2f} (Target Gap: {int(p_inner_width)}, Garment Outer: {int(g_outer_width)})")
        elif scale < min_scale * 0.9:
            scale = min_scale * 1.05
            print(f"  ⚠ Reverted to shoulder-based scale ({scale:.2f}) to prevent undersizing.")
            
        # Final Scale Guard
        scale = max(0.1, scale)

        # SLEEVE COVERAGE LOGIC (As requested)
        # Ensure the garment covers the person's existing shoulders/sleeves/arms
        # Focus ONLY on the "upper corners" by restricting the horizontal scan
        # Only apply this if the garment HAS sleeves (Sleeved mode)
        if has_g_sleeves:
            p_shoulder_y = int(min(p_pose['left_shoulder'][1], p_pose['right_shoulder'][1]))
            scan_y = p_shoulder_y + 10 # Scan slightly below shoulder landmarks
            if scan_y < person_img.shape[0]:
                # upper fashion classes for person - ensure sleeves (32) are included
                fashion_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 32] 
                p_upper_mask = np.isin(p_seg, fashion_ids).astype(np.uint8) * 255
                
                # Bare Arm Detection Fallback (B2) - As requested for sleeveless cases
                if p_seg_b2 is not None:
                    # 14: Left-arm, 15: Right-arm in B2
                    p_arms_mask = np.isin(p_seg_b2, [14, 15]).astype(np.uint8) * 255
                    p_upper_mask = cv2.bitwise_or(p_upper_mask, p_arms_mask)
                    
                row = p_upper_mask[scan_y, :]
                
                # CRITICAL REFINEMENT: Restrict horizontal scan to shoulder vicinity
                # This prevents picking up extended arms or flared sleeves
                p_l_s = p_pose['left_shoulder'][0]
                p_r_s = p_pose['right_shoulder'][0]
                p_center_x = (p_l_s + p_r_s) / 2.0
                p_torso_w = abs(p_l_s - p_r_s)
                
                # Scan only within 1.5x of the torso width around the center
                min_x = int(max(0, p_center_x - p_torso_w * 0.75))
                max_x = int(min(person_img.shape[1], p_center_x + p_torso_w * 0.75))
                
                row_restricted = row[min_x:max_x]
                coords = np.where(row_restricted > 0)[0]
                
                if len(coords) > 2:
                    p_outer_width = coords[-1] - coords[0]
                    g_outer_width = c_kp.get('outer_shoulder_width', g_t_width)
                    
                    coverage_scale = p_outer_width / (g_outer_width + 1e-6)
                    
                    if scale < coverage_scale * 1.02: # 2% safety margin
                        scale = coverage_scale * 1.05
                        print(f"  🛡️ Shoulder Corner Coverage Scaling: {scale:.2f} (Target Width: {int(p_outer_width)}, Garment Width: {int(g_outer_width)})")

        # 4. Project Landmarks onto Person Frame
        # Target Anchor: Above the shoulders (not at neckline)
        p_s_mid = (p_pose['left_shoulder'] + p_pose['right_shoulder']) / 2.0
        
        # Try to detect person's neckline from segmentation
        person_neckline_mask = (p_seg == 34).astype(np.uint8) * 255
        if np.sum(person_neckline_mask > 0) > 10:
            # Use center of person's neckline as reference
            M = cv2.moments(person_neckline_mask)
            if M["m00"] != 0:
                neck_ref = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)
                print(f"  🎯 Using person's segmented neckline (class 34) as reference")
        else:
            # Fallback: shoulder midpoint
            neck_ref = p_s_mid
        
        # Detect collar in cloth image (class 29)
        collar_mask = (c_seg == 29).astype(np.uint8) * 255
        has_collar = np.sum(collar_mask > 0) > 50
        
        # Lift the dress top ABOVE the shoulders
        # Use the minimum shoulder Y position and lift it further
        min_shoulder_y = min(p_pose['left_shoulder'][1], p_pose['right_shoulder'][1])
        
        if is_strapless_early:
            # CASE 4: STRAPLESS / TUBE GOWN (Refined User Request)
            # Align top edge exactly 2% below the person's shoulder level
            min_s_y = min(p_pose['left_shoulder'][1], p_pose['right_shoulder'][1])
            # Use image height for the 2% drop
            target_drop = person_img.shape[0] * 0.01
            target_anchor = np.array([p_s_mid[0], min_s_y + target_drop])
            print(f"  👗 Case 4: Strapless detected - aligning to 2% below shoulders.")
            
        elif is_case_3:
            # Case 3: Sleeveless - Top edge exactly on person's cloth top edge
            # Scan for absolute top Y of person's upper clothing in shoulder vicinity
            p_upper_mask = np.isin(p_seg, [1, 2, 3, 4, 5, 6, 11]).astype(np.uint8) * 255
            p_l_s = p_pose['left_shoulder'][0]
            p_r_s = p_pose['right_shoulder'][0]
            p_torso_w = abs(p_l_s - p_r_s)
            
            # Restrict scan to shoulder area
            min_x = int(max(0, min(p_l_s, p_r_s) - p_torso_w * 0.2))
            max_x = int(min(person_img.shape[1], max(p_l_s, p_r_s) + p_torso_w * 0.2))
            
            p_shoulder_crop = p_upper_mask[0:int(min_shoulder_y + p_torso_w*0.5), min_x:max_x]
            p_coords = cv2.findNonZero(p_shoulder_crop)
            if p_coords is not None:
                p_abs_top_y = np.min(p_coords[:, 0, 1])
                # Add 20% vertical drop BASED ON TORSO HEIGHT for significant seating (User requested 20%)
                vertical_drop = p_torso_height * 0.10
                target_anchor = np.array([p_s_mid[0], p_abs_top_y + vertical_drop])
                print(f"  🎽 Case 3 Alignment: Aligned to top edge (Y={p_abs_top_y}) with {int(vertical_drop)}px drop (20% height)")
            else:
                # Fallback with 20% drop
                vertical_drop = p_torso_height * 0.10
                target_anchor = np.array([p_s_mid[0], min_shoulder_y + vertical_drop])
                print(f"  🎽 Case 3 Alignment: Fallback with {int(vertical_drop)}px drop")
        else:
            if has_collar:
                lift_amount = p_t_width * 0.33  # More lift for collared garments
                print(f"  👔 Collar detected - using 30% lift")
            else:
                lift_amount = p_t_width * 0.04  # Standard lift for regular garments
            
            target_anchor = np.array([p_s_mid[0], min_shoulder_y - lift_amount])
        
        print(f"  📍 Target Anchor: ({int(target_anchor[0])}, {int(target_anchor[1])})")
        print(f"  📍 Garment Anchor: ({int(g_anchor[0])}, {int(g_anchor[1])})")
        
        # Project source points to target space using unified scale
        projected_p_kp = {}
        for name, s_pt in c_kp.items():
            if not isinstance(s_pt, np.ndarray) or s_pt.ndim != 1:
                continue
            dx = (s_pt[0] - g_anchor[0])
            dy = (s_pt[1] - g_anchor[1])
            # Unified proportional scaling (zero stretching)
            p_pos = target_anchor + np.array([dx * scale, dy * scale])
            projected_p_kp[name] = p_pos

        # Blend limbs (elbows/wrists) with actual person pose
        # We use a very high weight for person pose to "guide" the scaled sleeves onto the arms
        blend_kps = ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        for name in blend_kps:
            if name in projected_p_kp and name in p_pose:
                # 0.9 weight - enough to move them, but keep texture rigid
                projected_p_kp[name] = (projected_p_kp[name] * 0.1) + (p_pose[name] * 0.9)

        # 5. Simple Scale + Translate with Length Limit
        # Detect person's ankle position for max dress length (Feet level)
        if 'left_ankle' in p_pose and 'right_ankle' in p_pose:
            avg_ankle_y = (p_pose['left_ankle'][1] + p_pose['right_ankle'][1]) / 2.0
            if is_strapless_early:
                # Strapless target: EXACTLY at feet level (approx 7.5% below ankles to reach floor)
                max_dress_bottom = avg_ankle_y + (person_img.shape[0] * 0.075) 
                print(f"  👗 Strapless Length: Targeting Feet/Floor level.")
            else:
                max_dress_bottom = avg_ankle_y + (person_img.shape[0] * 0.03)  # 3% below feet standard
        else:
            # Fallback: use image height
            max_dress_bottom = person_img.shape[0]
        
        # Step 1: Calculate initial scaled dimensions
        new_w = int(bw * scale)
        new_h = int(bh * scale)
        
        # Check if dress would extend too far below feet
        scaled_g_anchor = (g_anchor - np.array([bx, by])) * scale
        projected_bottom = target_anchor[1] + (new_h - scaled_g_anchor[1])
        
        if projected_bottom > max_dress_bottom:
            # Calculate how much we need to reduce the height
            allowed_height = max_dress_bottom - target_anchor[1] + scaled_g_anchor[1]
            height_ratio = allowed_height / new_h
            
            # Reduce BOTH width and height proportionally to maintain aspect ratio
            # But wait, we need to scroll down to find the end of process first.
            # I will use view_file again to find the end of process before replacing.

            new_h = int(allowed_height)
            new_w = int(new_w * height_ratio)
            scale = scale * height_ratio  # Update scale for both dimensions
            
            print(f"  ✂️ Trimmed dress proportionally to fit ankle+5% limit (aspect ratio preserved)")
        
        # Resize garment
        cutout = cv2.bitwise_and(cloth_img[by:by+bh, bx:bx+bw], cloth_img[by:by+bh, bx:bx+bw], mask=c_mask[by:by+bh, bx:bx+bw])
        mask_cutout = c_mask[by:by+bh, bx:bx+bw]
        
        scaled_cloth = cv2.resize(cutout, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        scaled_mask = cv2.resize(mask_cutout, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Step 2: Calculate translation to place garment anchor at target anchor
        scaled_g_anchor = (g_anchor - np.array([bx, by])) * scale
        
        tx = target_anchor[0] - scaled_g_anchor[0]
        ty = target_anchor[1] - scaled_g_anchor[1]
        
        # Build translation matrix
        M_translate = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        
        # Apply translation
        ph, pw = person_img.shape[:2]
        warped_cloth = cv2.warpAffine(scaled_cloth, M_translate, (pw, ph), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_mask = cv2.warpAffine(scaled_mask, M_translate, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if warped_cloth is not None:
            # Finishing and Compositing
            if is_case_3:
                # Targeted Shoulder Trimming: Clip only the TOP parts (straps) to the body silhouette
                # Use ALL person parts (p_seg_b2 > 0) to ensure neck is NOT clipped
                body_silhouette = (p_seg_b2 > 0).astype(np.uint8) * 255
                
                # Identify the "Shoulder Zone" of the garment (top 20% height above target_anchor)
                coords_g = cv2.findNonZero(warped_mask)
                if coords_g is not None:
                    gy_top = np.min(coords_g[:, 0, 1])
                    gy_bottom = np.max(coords_g[:, 0, 1])
                    g_height = gy_bottom - gy_top
                    # Only trim the top 25% of the garment (shoulder area)
                    shoulder_trim_y = int(gy_top + g_height * 0.25)
                    
                    # Create a mask that is only active in the shoulder zone
                    shoulder_zone = np.zeros_like(warped_mask)
                    shoulder_zone[0:shoulder_trim_y, :] = 255
                    
                    # Clip the warped mask to body silhouette ONLY in the shoulder zone
                    clipped_top = cv2.bitwise_and(warped_mask, body_silhouette)
                    
                    # Keep clipped top in shoulder zone, and original mask elsewhere
                    warped_mask = np.where(shoulder_zone > 0, clipped_top, warped_mask)
                    print(f"  ✂️ Targeted Shoulder Trimming applied to Case 3 (Zone Threshold Y={shoulder_trim_y})")

            # STOP using B2 labels (2, 11, 13) for exclusion - Protect ONLY the face using MediaPipe
            exclusion = np.zeros_like(p_seg_b2, dtype=np.uint8)
            if 'face_pts' in p_pose:
                hull = cv2.convexHull(np.array(p_pose['face_pts'], np.int32))
                cv2.fillPoly(exclusion, [hull], 255)
            
            # Remove from garment anything that overlaps the protected FACE area
            warped_mask = cv2.bitwise_and(warped_mask, cv2.bitwise_not(exclusion))
            
            # Define target for unified mask generation
            target_original = original_img if original_img is not None else person_img
            
            alpha = (cv2.GaussianBlur(warped_mask, (7,7), 0).astype(float) / 255.0)[..., None]
            # --- UNIFIED MASK GENERATION ---
            print("  🎭 Generating Unified VTON Mask...")
            
            unified_mask = self.masker.get_final_mask(target_original, warped_mask, mode='universal', clean_img=person_clean)
            
            # --- USER REQUEST: Expand mask interior for Strapless (Bottom 40%) ---
            if is_strapless_early:
                print("  🎭 Case 4: Expanding bottom 40% interior in FINAL_MASK...")
                _, cloth_bin = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
                ys, xs = np.where(cloth_bin > 0)
                if len(ys) > 0:
                    y_min_g, y_max_g = np.min(ys), np.max(ys)
                    total_h_g = y_max_g - y_min_g
                    # Define bottom 40% region
                    b40_y_start = y_max_g - int(total_h_g * 0.40)
                    
                    # Create a mask for this vertical strip
                    b40_strip = np.zeros_like(unified_mask)
                    b40_strip[max(0, b40_y_start):min(ph, y_max_g), :] = 255
                    
                    # Extract garment interior within this strip
                    b40_interior = cv2.bitwise_and(cloth_bin, b40_strip)
                    
                    # Incorporate into unified mask
                    unified_mask = cv2.bitwise_or(unified_mask, b40_interior)
            # ----------------------------------------------------------------------
            
            # Save Mask
            mask_out = out_dir / "final_mask.png"
            cv2.imwrite(str(mask_out), unified_mask)
            print(f"✅ Saved Unified Mask to: {mask_out}")

            result = (warped_cloth * alpha + person_clean * (1 - alpha)).astype(np.uint8)
            
            # --- ARTIFICIAL SKIN HELPER (Combined Logic) ---
            print("  🦴 Drawing Artificial Skin Skeleton...")
            
            # 1. Determine garment categories for limb rendering (Standard B3 Classes)
            UPPER_CLASSES = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 28, 29, 32, 34, 36} | set(range(37, 47))
            LOWER_CLASSES = {7, 8, 9, 20}
            
            has_upper = np.any(np.isin(c_seg, list(UPPER_CLASSES)))
            has_lower = np.any(np.isin(c_seg, list(LOWER_CLASSES)))
            
            # USER: Draw legs for dresses/mini-dresses too
            FULL_LENGTH_SHOW_LEGS = {11, 12, 13, 20} # Dress, Jumpsuit, etc.
            has_full_length = np.any(np.isin(c_seg, list(FULL_LENGTH_SHOW_LEGS)))
            
            # Render legs if it's a lower garment OR a show-leg full-length garment, but NOT if pants are the primary feature
            draw_legs = (has_lower or has_full_length) and not has_pants
            
            # 2. Process Skin
            # PASS upper_body_mask=None for universal mode because we are replacing ALL clothes
            # This prevents original clothes from clipping the new artificial limbs.
            result, skin_mask = self.skin_helper.process(
                result, warped_mask, out_dir, pose_kps=p_pose,
                include_arms=has_upper, include_legs=draw_legs,
                sampling_img=target_original, upper_body_mask=None
            )
            
            # USER REQUEST: Expand artificial arm mask ONLY in the final mask
            print("  🎭 Expanding Artificial Skin Mask for FINAL_MASK...")
            k_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
            expanded_skin_mask = cv2.dilate(skin_mask, k_skin, iterations=1)
            
            # Add expanded artificial skin mask to the unified mask
            combined_mask = cv2.bitwise_or(unified_mask, expanded_skin_mask)

            # --- FINAL FACE PROTECTION (Override) ---
            print("  🛡️ Applying Final Face Shield...")
            face_shield = self.masker.get_face_shield(target_original)
            combined_mask[face_shield > 0] = 0

            cv2.imwrite(str(mask_out), combined_mask)
            print(f"✅ Updated final_mask with expanded artificial skin")
            
            cv2.imwrite(str(out_dir / "final_result.png"), result)
            print(f"✓ Saved: {out_dir / 'final_result.png'}")
            
        return result

    def estimate_source_pose_mask_based(self, mask, bbox, g_anchor_pt=None, c_seg=None):
        """Segmentation-based garment analysis using class 34 (neckline) for precision"""
        x, y, w, h = bbox
        
        # 1. PRECISION NECKLINE DETECTION using class 34 segmentation
        if c_seg is not None:
            # Extract neckline (class 34)
            neckline_mask = (c_seg == 34).astype(np.uint8) * 255
            if np.sum(neckline_mask > 0) > 10:
                # Find center of the neckline
                M = cv2.moments(neckline_mask)
                if M["m00"] != 0:
                    neck_x = M["m10"] / M["m00"]
                    neck_y = M["m01"] / M["m00"]
                    g_anchor_pt = np.array([neck_x, neck_y], dtype=np.float32)
                    print(f"  🎯 Using segmented neckline (class 34) as anchor")
        
        # Fallback: Neck-centric alignment
        if g_anchor_pt is None:
            y_neck_detect = int(y + h * 0.1)
            neck_row = mask[y_neck_detect, :]
            neck_coords = np.where(neck_row > 0)[0]
            
            if len(neck_coords) > 2:
                c_neck_center_x = (neck_coords[0] + neck_coords[-1]) / 2.0
            else:
                c_neck_center_x = x + w / 2.0
            
            # Find actual top y
            cloth_top_y = y
            for cy in range(y, y + int(h * 0.3)):
                if np.any(mask[cy, :] > 0):
                    cloth_top_y = cy
                    break
            
            g_anchor_pt = np.array([c_neck_center_x, cloth_top_y], dtype=np.float32)
        
        c_neck_center_x = g_anchor_pt[0]
        kp = {'neck_anchor': g_anchor_pt}
        
        # 2. DUPATTA GUARD: Symmetric Torso Width Detection
        y_scan = int(y + h * 0.15)
        row_scan = mask[y_scan, :]
        scan_coords = np.where(row_scan > 0)[0]
        
        if len(scan_coords) > 2:
            dist_l = max(0, c_neck_center_x - scan_coords[0])
            dist_r = max(0, scan_coords[-1] - c_neck_center_x)
            
            # Dupatta detection
            if max(dist_l, dist_r) > min(dist_l, dist_r) * 1.4 and min(dist_l, dist_r) > 10:
                shoulder_width_estimate = 2 * min(dist_l, dist_r)
                print(f"  🧣 Dupatta detected! Using symmetric core width: {int(shoulder_width_estimate)}")
            else:
                shoulder_width_estimate = dist_l + dist_r
                
            # Calculate shoulders
            kp['left_shoulder'] = np.array([c_neck_center_x + shoulder_width_estimate * 0.45, y + h*0.18], dtype=np.float32)
            kp['right_shoulder'] = np.array([c_neck_center_x - shoulder_width_estimate * 0.45, y + h*0.18], dtype=np.float32)
        else:
            shoulder_width_estimate = w * 0.6
            kp['left_shoulder'] = np.array([c_neck_center_x + shoulder_width_estimate * 0.5, y + h*0.18], dtype=np.float32)
            kp['right_shoulder'] = np.array([c_neck_center_x - shoulder_width_estimate * 0.5, y + h*0.18], dtype=np.float32)
        
        # STRAPLESS DETECTION (Tube gowns/tops)
        has_g_sleeves = (c_seg is not None and np.sum(c_seg == 32) > 100)
        has_g_collar = (c_seg is not None and np.sum(c_seg == 29) > 100)
        has_g_neckline = (c_seg is not None and np.sum(c_seg == 34) > 100)
        
        is_strapless = (not has_g_sleeves and not has_g_collar and not has_g_neckline)
        if is_strapless:
            # Re-align neck_anchor to be the top center of the detected garment body
            kp['neck_anchor'] = np.array([c_neck_center_x, cloth_top_y], dtype=np.float32)
            kp['is_strapless'] = True
            print("  👗 STRAPLESS case flagged.")
        
        # 3. SLEEVE DETECTION using class 32 (sleeve) if available
        # First, find the absolute furthest extremities from full mask as reference
        coords_all = cv2.findNonZero(mask)
        if coords_all is not None:
            pts_all = coords_all.reshape(-1, 2)
            mask_left_tip = pts_all[np.argmax(pts_all[:, 0])].astype(np.float32)
            mask_right_tip = pts_all[np.argmin(pts_all[:, 0])].astype(np.float32)
            mask_width = mask_left_tip[0] - mask_right_tip[0]
        else:
            mask_left_tip = np.array([x + w, y + h*0.5], dtype=np.float32)
            mask_right_tip = np.array([x, y + h*0.5], dtype=np.float32)
            mask_width = w
        
        sleeve_validated = False
        if c_seg is not None:
            sleeve_mask = (c_seg == 32).astype(np.uint8) * 255
            if np.sum(sleeve_mask > 0) > 100:
                # Find left and right sleeve tips
                coords = cv2.findNonZero(sleeve_mask)
                if coords is not None:
                    pts = coords.reshape(-1, 2)
                    l_tip = pts[np.argmax(pts[:, 0])].astype(np.float32)
                    r_tip = pts[np.argmin(pts[:, 0])].astype(np.float32)
                    sleeve_width = l_tip[0] - r_tip[0]
                    
                    # Validate: sleeve width should be at least 60% of total mask width
                    # This prevents false detections from dupatta decorative elements
                    if sleeve_width >= mask_width * 0.6:
                        kp['left_wrist'] = l_tip
                        kp['right_wrist'] = r_tip
                        sleeve_validated = True
                        print(f"  👔 Using segmented sleeves (class 32) for wrist positions")
                    else:
                        print(f"  ⚠ Class 32 sleeve span too narrow ({int(sleeve_width)}px vs {int(mask_width)}px), using full mask width")
        
        # Fallback or if validation failed: Use full mask extremities
        if not sleeve_validated:
            kp['left_wrist'] = mask_left_tip
            kp['right_wrist'] = mask_right_tip

        kp['left_elbow'] = (kp['left_shoulder'] + kp['left_wrist']) / 2
        kp['right_elbow'] = (kp['right_shoulder'] + kp['right_wrist']) / 2
        
        # 4. Torso & Bottom
        kp['torso_mid'] = np.array([c_neck_center_x, y + h*0.5], dtype=np.float32)
        kp['left_hip'] = np.array([c_neck_center_x + w*0.3, y + h*0.8], dtype=np.float32)
        kp['right_hip'] = np.array([c_neck_center_x - w*0.3, y + h*0.8], dtype=np.float32)
        kp['bottom_center'] = np.array([c_neck_center_x, y + h], dtype=np.float32)
        
        # Save the full width for coverage-aware scaling
        kp['outer_shoulder_width'] = shoulder_width_estimate
        
        return kp

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Universal Garment Warper (TPS/LBS)")
    parser.add_argument("--person", type=str, default="person.jpg", help="Path to person image")
    parser.add_argument("--cloth", type=str, default="cloth.png", help="Path to cloth image")
    parser.add_argument("--out", type=str, default="./output_test", help="Output directory")
    args = parser.parse_args()

    # Discover project root
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Standardized paths from models_downloader.py
    B2_MODEL = base_dir / "models" / "segformer_b2_clothes.onnx"
    B3_MODEL = base_dir / "models" / "segformer-b3-fashion"
    
    # Check if models exist
    if not B2_MODEL.exists():
        print(f"❌ Error: B2 model not found at {B2_MODEL}")
        return
    if not B3_MODEL.exists():
        print(f"❌ Error: B3 model not found at {B3_MODEL}")
        return
    
    print(f"--- Universal Warper Start ---")
    print(f"Person: {args.person}")
    print(f"Cloth:  {args.cloth}")
    
    warper = UniversalGarmentWarper(B2_MODEL, B3_MODEL)
    p_img = cv2.imread(args.person)
    c_img = cv2.imread(args.cloth)
    
    if p_img is None: return print(f"✗ Failed to load person image: {args.person}")
    if c_img is None: return print(f"✗ Failed to load cloth image: {args.cloth}")
    
    warper.process(p_img, c_img, args.out)

if __name__ == "__main__":
    main()
