import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import os
import sys

# Set protocol buffers implementation to python to avoid issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from modules.virtual_tryon2.vton_masking_helper import VTONMasker
from modules.virtual_tryon2.vton_skin_helper import VTONSkinHelper

class UniversalGarmentWarper:
    def __init__(self, b2_model_path, b3_model_path):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.masker = VTONMasker(str(b2_model_path), str(b3_model_path))
        self.skin_helper = VTONSkinHelper()
        self.b2_model_path = b2_model_path
        self.b3_model_path = b3_model_path

    def segment_b2(self, image):
        return self.masker.get_seg_map(image, model_type='b2')

    def segment_b3(self, image):
        return self.masker.get_seg_map(image, model_type='b3')

    def detect_pose(self, image, b3_seg=None):
        """Standard MediaPipe pose detection with segmentation fallback"""
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            H, W = image.shape[:2]
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Face points for exclusion
                face_pts = []
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: # Nose, Eyes, Ears
                    face_pts.append([landmarks[idx].x * W, landmarks[idx].y * H])
                
                return {
                    'left_shoulder': np.array([landmarks[11].x * W, landmarks[11].y * H]),
                    'right_shoulder': np.array([landmarks[12].x * W, landmarks[12].y * H]),
                    'left_elbow': np.array([landmarks[13].x * W, landmarks[13].y * H]),
                    'right_elbow': np.array([landmarks[14].x * W, landmarks[14].y * H]),
                    'left_wrist': np.array([landmarks[15].x * W, landmarks[15].y * H]),
                    'right_wrist': np.array([landmarks[16].x * W, landmarks[16].y * H]),
                    'left_hip': np.array([landmarks[23].x * W, landmarks[23].y * H]),
                    'right_hip': np.array([landmarks[24].x * W, landmarks[24].y * H]),
                    'left_knee': np.array([landmarks[25].x * W, landmarks[25].y * H]),
                    'right_knee': np.array([landmarks[26].x * W, landmarks[26].y * H]),
                    'left_ankle': np.array([landmarks[27].x * W, landmarks[27].y * H]),
                    'right_ankle': np.array([landmarks[28].x * W, landmarks[28].y * H]),
                    'face_pts': face_pts
                }
        
        # Fallback: Estimate from Segmentation
        print("  Using segmentation-based pose estimation for person...")
        if b3_seg is None:
            b3_seg = self.segment_b3(image)
        
        return self._estimate_person_pose_from_seg(b3_seg, H, W)

    def _estimate_person_pose_from_seg(self, seg, H, W):
        """Estimates keypoints by analyzing the person's segmentation mask"""
        upper_mask = np.isin(seg, [1, 2, 3, 4, 5, 6, 10, 11, 12]).astype(np.uint8) * 255
        coords = cv2.findNonZero(upper_mask)
        if coords is None: return None
        
        x, y, w, h = cv2.boundingRect(coords)
        kp = {
            'left_shoulder': np.array([x + w*0.8, y + h*0.2], dtype=np.float32),
            'right_shoulder': np.array([x + w*0.2, y + h*0.2], dtype=np.float32),
            'left_hip': np.array([x + w*0.75, y + h*0.9], dtype=np.float32),
            'right_hip': np.array([x + w*0.25, y + h*0.9], dtype=np.float32),
        }
        return kp

    def get_tps_warp(self, cutout, mask, src_kp, dst_kp, target_shape, bbox):
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
        s_pts_arr = np.array(s_pts).reshape(1, -1, 2).astype(np.float32)
        d_pts_arr = np.array(d_pts).reshape(1, -1, 2).astype(np.float32)
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

    def get_lbs_warp(self, pants_img, src_pose, dst_pose, target_shape, mask=None):
        h, w = target_shape[:2]
        bones_list = ['waist', 'l_thigh', 'l_shin', 'r_thigh', 'r_shin']
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
        weights_map = self._compute_skinning_weights((h, w), dst_pose, bones_list)
        y, x = np.mgrid[0:h, 0:w]
        P_dst = np.stack([x, y, np.ones_like(x)], axis=0).reshape(3, -1)
        P_src_acc = np.zeros((2, h*w), dtype=np.float32)
        for i, b_name in enumerate(bones_list):
            if b_name in transforms:
                P_src_acc += (transforms[b_name] @ P_dst) * weights_map[:, :, i].reshape(-1)
        mx, my = P_src_acc[0, :].reshape(h, w), P_src_acc[1, :].reshape(h, w)
        warped_img = cv2.remap(pants_img, mx, my, cv2.INTER_LANCZOS4, borderValue=(0,0,0))
        warped_mask = cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0) if mask is not None else None
        return warped_img, warped_mask

    def _get_bones(self, p):
        return {
            'waist': (p['left_hip'], p['right_hip']),
            'l_thigh': (p['left_hip'], p.get('left_knee', p['left_hip']+[0,100])), 
            'l_shin': (p.get('left_knee', p['left_hip']+[0,100]), p.get('left_ankle', p['left_hip']+[0,200])),
            'r_thigh': (p['right_hip'], p.get('right_knee', p['right_hip']+[0,100])), 
            'r_shin': (p.get('right_knee', p['right_hip']+[0,100]), p.get('right_ankle', p['right_hip']+[0,200]))
        }

    def _compute_aniso_affine(self, s1, s2, d1, d2, width_scale):
        sv, dv = s2 - s1, d2 - d1
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
            if lab2 < 1e-6: d = np.linalg.norm(pts - a, axis=1)
            else:
                t = np.clip(np.sum((pts - a) * ab, axis=1) / lab2, 0, 1)
                d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
            weights[:, i] = -(d**2) / (2 * sigma**2)
        row_max = np.max(weights, axis=1, keepdims=True)
        exp_w = np.exp(weights - row_max)
        weights = exp_w / np.sum(exp_w, axis=1, keepdims=True)
        weights = weights.reshape(sh, sw, len(bones_list))
        full_w = [cv2.resize(weights[:,:,i], (w, h)) for i in range(len(bones_list))]
        return np.stack(full_w, axis=2)

    def process(self, person_img, cloth_img, out_dir, original_img=None, clean_img=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        print("Segmenting person & cloth...")
        analysis_img = original_img if original_img is not None else person_img
        p_seg = self.segment_b3(analysis_img)
        c_seg = self.segment_b3(cloth_img)
        p_seg_b2 = self.segment_b2(analysis_img)
        
        if clean_img is not None: person_clean = clean_img
        else:
            try:
                from modules.virtual_tryon2.cloth_remover import ClothRemover
                remover_model_path = self.base_dir / "models" / "SegFormerB2Clothes" / "segformer_b2_clothes.onnx"
                if not remover_model_path.exists(): remover_model_path = self.base_dir / "models" / "segformer_b2_clothes.onnx"
                remover = ClothRemover(str(remover_model_path))
                person_clean, _ = remover.remove_all(person_img)
            except Exception: person_clean = person_img.copy()

        p_pose = self.detect_pose(analysis_img, p_seg)
        if not p_pose: return None
        
        fashion_ids = list(range(1, 47))
        c_mask = np.isin(c_seg, fashion_ids).astype(np.uint8) * 255
        if np.sum(c_mask > 0) < 500: return None
        
        # Automatic Garment Type Detection
        UPPER_CLASSES = {1, 2, 3, 4, 5, 6, 10, 28, 29, 32, 34, 39, 40, 41, 42, 43, 44}
        LOWER_CLASSES = {7, 8, 9, 33}
        upper_px = np.sum(np.isin(c_seg, list(UPPER_CLASSES)))
        lower_px = np.sum(np.isin(c_seg, list(LOWER_CLASSES)))
        g_type = 'lower' if lower_px > upper_px * 1.5 else 'upper'
        if np.sum(np.isin(c_seg, [11, 12, 13])) > 5000: g_type = 'overall'
        print(f"  [Detection] Garment Type: {g_type.upper()}")
        
        coords = cv2.findNonZero(c_mask)
        c_bbox = cv2.boundingRect(coords)
        bx, by, bw, bh = c_bbox
        c_kp = self.estimate_source_pose_mask_based(c_mask, c_bbox, c_seg=c_seg)
        g_anchor = c_kp['neck_anchor']
        
        # Scaling Refinement
        p_span = np.linalg.norm(p_pose.get('left_wrist', p_pose['left_shoulder']) - p_pose.get('right_wrist', p_pose['right_shoulder']))
        g_span = np.linalg.norm(c_kp['left_wrist'] - c_kp['right_wrist'])
        scale_width = p_span / (g_span + 1e-6)
        
        has_sequins = np.sum(c_seg == 45) > 500
        if (g_type == 'lower' or has_sequins) and 'left_ankle' in p_pose and 'right_ankle' in p_pose:
            p_min_s_y = min(p_pose['left_shoulder'][1], p_pose['right_shoulder'][1])
            p_avg_a_y = (p_pose['left_ankle'][1] + p_pose['right_ankle'][1]) / 2.0
            p_target_h = (p_avg_a_y + (person_img.shape[0]*0.05 if has_sequins else 0)) - p_min_s_y
            g_source_h = (by + bh) - g_anchor[1]
            scale = p_target_h / (g_source_h + 1e-6)
        else:
            scale = scale_width

        # Warping
        if g_type == 'lower':
            warped_cloth, warped_mask = self.get_lbs_warp(cloth_img, c_kp, p_pose, person_img.shape, mask=c_mask)
        else:
            for k in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                if k in p_pose: p_pose[k] = p_pose[k]*0.9 + (p_pose['left_shoulder'] if 'left' in k else p_pose['right_shoulder'])*0.1
            warped_cloth, warped_mask = self.get_tps_warp(cloth_img, c_mask, c_kp, p_pose, person_img.shape, c_bbox)

        if warped_cloth is None: return None

        # Masking & Protection
        ex = np.zeros(p_seg.shape, np.uint8)
        if 'face_pts' in p_pose: cv2.fillPoly(ex, [cv2.convexHull(np.array(p_pose['face_pts'], np.int32))], 255)
        warped_mask = cv2.bitwise_and(warped_mask, cv2.bitwise_not(ex))
        if g_type == 'lower':
            upper_cloth_mask = np.isin(p_seg, [1, 2, 3, 4, 5, 6, 10]).astype(np.uint8) * 255
            warped_mask = cv2.bitwise_and(warped_mask, cv2.bitwise_not(upper_cloth_mask))
        
        try:
            h_mask = self.masker.get_hand_mask(analysis_img)
            if h_mask is not None: warped_mask = cv2.bitwise_and(warped_mask, cv2.bitwise_not(h_mask))
        except: pass

        alpha = (cv2.GaussianBlur(warped_mask, (7,7), 0).astype(float) / 255.0)[..., None]
        result = (warped_cloth * alpha + person_clean * (1 - alpha)).astype(np.uint8)
        
        unified_mask = self.masker.get_final_mask(analysis_img, warped_mask, mode='universal', clean_img=person_clean)
        result, skin_mask = self.skin_helper.process(result, warped_mask, out_dir, pose_kps=p_pose, 
                                                    include_arms=(g_type!='lower'), include_legs=(g_type in ['lower','overall']),
                                                    sampling_img=analysis_img, upper_body_mask=None)
        
        final_mask = cv2.bitwise_or(unified_mask, cv2.dilate(skin_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41,41))))
        face_shield = self.masker.get_face_shield(analysis_img)
        final_mask[face_shield > 0] = 0

        cv2.imwrite(str(out_dir / "final_mask.png"), final_mask)
        cv2.imwrite(str(out_dir / "final_result.png"), result)
        return result

    def estimate_source_pose_mask_based(self, mask, bbox, g_anchor_pt=None, c_seg=None):
        x, y, w, h = bbox
        if c_seg is not None:
            neck_mask = (c_seg == 34).astype(np.uint8) * 255
            if np.sum(neck_mask > 0) > 10:
                M = cv2.moments(neck_mask)
                if M["m00"] != 0: g_anchor_pt = np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]], dtype=np.float32)
        
        if g_anchor_pt is None:
            y_neck = int(y + h * 0.1)
            row = mask[y_neck, :]
            coords = np.where(row > 0)[0]
            cx = (coords[0] + coords[-1]) / 2.0 if len(coords) > 2 else x + w/2.0
            g_anchor_pt = np.array([cx, y], dtype=np.float32)
        
        cx = g_anchor_pt[0]
        kp = {'neck_anchor': g_anchor_pt, 'left_shoulder': np.array([cx+w*0.35, y+h*0.15]), 'right_shoulder': np.array([cx-w*0.35, y+h*0.15])}
        kp['left_wrist'] = np.array([cx+w*0.45, y+h*0.6])
        kp['right_wrist'] = np.array([cx-w*0.45, y+h*0.6])
        kp['left_hip'] = np.array([cx+w*0.3, y+h*0.8])
        kp['right_hip'] = np.array([cx-w*0.3, y+h*0.8])
        return kp

def main():
    base_dir = Path(__file__).resolve().parents[2]
    B2_MODEL = base_dir / "models" / "SegFormerB2Clothes" / "segformer_b2_clothes.onnx"
    if not B2_MODEL.exists(): B2_MODEL = base_dir / "models" / "segformer_b2_clothes.onnx"
    B3_MODEL = base_dir / "models" / "segformer-b3-fashion"
    if not B2_MODEL.exists() or not B3_MODEL.exists(): return print("Models missing")
    print("VTON Logic Finalized.")

if __name__ == "__main__":
    main()
