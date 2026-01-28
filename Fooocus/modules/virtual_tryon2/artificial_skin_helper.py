import cv2
import numpy as np
from pathlib import Path
from .model_loader import get_pose_detector

class ArtificialSkinHelper:
    def __init__(self):
        self.pose = get_pose_detector()
        # Default skin color (Peach/Tan)
        self.default_skin_color = (138, 176, 245) 

    def estimate_skin_color(self, image, pose_kps):
        """Samples skin color from the face region, favoring lighter shades"""
        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # 1. Sample from Face (Most reliable skin source)
        face_pts = pose_kps.get('face_pts', [])
        if len(face_pts) > 1:
            pts = np.array(face_pts, np.int32)
            cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)
        elif 'nose' in pose_kps:
            # MediaPipe points: 0=nose, 1=left_eye_inner, 4=right_eye_inner (Legacy fallback)
            cv2.circle(mask, tuple(pose_kps['nose'].astype(int)), int(W * 0.02), 255, -1)
            
        if np.sum(mask > 0) < 10:
            # Fallback to a wider nose-center area if face_pts missing
            if 'nose' in pose_kps:
                cv2.circle(mask, tuple(pose_kps['nose'].astype(int)), int(W * 0.04), 255, -1)
            else:
                return self.default_skin_color
        
        # Clean mask: Erode to avoid background contamination (hair, background)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
            
        pixels = image[mask > 0]
        if len(pixels) < 10:
            return self.default_skin_color
        
        # USER REQUEST: "light shade not dark" & "get eh shade from face"
        # We use 75th percentile for brightness but keep it within skin range
        light_color = np.percentile(pixels, 75, axis=0).astype(int)
        
        return tuple(map(int, light_color))

    def detect_pose(self, image):
        H, W = image.shape[:2]
        res = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        
        lms = res.pose_landmarks.landmark
        def to_px(idx): return np.array([lms[idx].x * W, lms[idx].y * H], dtype=np.float32)
        
        return {
            'left_shoulder': to_px(11), 'right_shoulder': to_px(12),
            'left_elbow': to_px(13), 'right_elbow': to_px(14),
            'left_wrist': to_px(15), 'right_wrist': to_px(16),
            'left_hip': to_px(23), 'right_hip': to_px(24),
            'left_knee': to_px(25), 'right_knee': to_px(26),
            'left_ankle': to_px(27), 'right_ankle': to_px(28),
            'left_pinky': to_px(17), 'right_pinky': to_px(18),
            'left_index': to_px(19), 'right_index': to_px(20),
            'left_thumb': to_px(21), 'right_thumb': to_px(22),
            'left_heel': to_px(29), 'right_heel': to_px(30),
            'left_foot_index': to_px(31), 'right_foot_index': to_px(32)
        }

    def draw_limb(self, mask, start, end, thickness):
        """Draws a thick line representing a limb segment"""
        p1 = tuple(np.array(start).astype(int))
        p2 = tuple(np.array(end).astype(int))
        cv2.line(mask, p1, p2, 255, thickness)

    def generate_skin_mask(self, person_img, garment_mask, pose_kps, include_arms=True, include_legs=True, upper_body_mask=None):
        H, W = person_img.shape[:2]
        
        # Ensure garment mask matches skin mask size
        if garment_mask.shape[:2] != (H, W):
            garment_mask = cv2.resize(garment_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        combined_mask = np.zeros((H, W), dtype=np.uint8)

        # 1. DRAW LEGS
        if include_legs:
            leg_mask = np.zeros((H, W), dtype=np.uint8)
            t = int(W * 0.055) # Reverted to original thickness
            
            segments = [
                ('left_hip', 'left_knee', t), ('left_knee', 'left_ankle', t),
                ('right_hip', 'right_knee', t), ('right_knee', 'right_ankle', t)
            ]
            
            for s, e, thick in segments:
                if s in pose_kps and e in pose_kps:
                    self.draw_limb(leg_mask, pose_kps[s], pose_kps[e], thick)
            
            # Mask legs by garment (So they only show at the bottom)
            # Increase kernel slightly to ensure NO overlap at the seam
            dilated_garment = cv2.dilate(garment_mask, np.ones((5,5), np.uint8), iterations=1)
            leg_mask = cv2.bitwise_and(leg_mask, cv2.bitwise_not(dilated_garment))
            
            # USER REQUEST: Ensure legs don't hide BEHIND upper body garment either
            if upper_body_mask is not None:
                leg_mask = cv2.bitwise_and(leg_mask, cv2.bitwise_not(upper_body_mask))

            combined_mask = cv2.bitwise_or(combined_mask, leg_mask)
        
        # 2. DRAW ARMS (Sleeve-Aware Placement)
        if include_arms:
            t_arm = int(W * 0.05)
            t_hand = int(W * 0.03)
            
            # Identify Torso and Sleeve regions of the garment
            torso_hull_mask = np.zeros((H, W), dtype=np.uint8)
            torso_pts_list = []
            for kp in ['left_shoulder', 'right_shoulder', 'right_hip', 'left_hip']:
                if kp in pose_kps:
                    torso_pts_list.append(pose_kps[kp])
            
            if len(torso_pts_list) >= 4:
                pts = np.array(torso_pts_list, np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillPoly(torso_hull_mask, [hull], 255)
                # Dilation to cover the loose shirt body
                kernel_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                torso_hull_mask = cv2.dilate(torso_hull_mask, kernel_t, iterations=1)
            
            sleeve_mask = cv2.bitwise_and(garment_mask, cv2.bitwise_not(torso_hull_mask))
            
            # Define segments
            upper_segments = [
                ('left_shoulder', 'left_elbow', t_arm), ('right_shoulder', 'right_elbow', t_arm)
            ]
            lower_segments = [
                ('left_elbow', 'left_wrist', t_arm), ('right_elbow', 'right_wrist', t_arm),
                ('left_wrist', 'left_pinky', t_hand), ('left_wrist', 'left_index', t_hand), ('left_wrist', 'left_thumb', t_hand),
                ('right_wrist', 'right_pinky', t_hand), ('right_wrist', 'right_index', t_hand), ('right_wrist', 'right_thumb', t_hand)
            ]
            
            # --- JOINT PIVOTS: Ensure continuity at joints ---
            pivot_mask = np.zeros((H, W), dtype=np.uint8)
            for kp in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                if kp in pose_kps:
                    pt = tuple(np.array(pose_kps[kp]).astype(int))
                    cv2.circle(pivot_mask, pt, int(t_arm * 0.55), 255, -1)
            
            # --- UPPER ARMS: Masked by TOTAL garment ---
            # User: "hidden behind both the shirt and the sleeves"
            u_mask = np.zeros((H, W), dtype=np.uint8)
            for s, e, t in upper_segments:
                if s in pose_kps and e in pose_kps:
                    self.draw_limb(u_mask, pose_kps[s], pose_kps[e], t)
            
            # Add elbow pivot to upper mask to smooth the bend
            for elbow in ['left_elbow', 'right_elbow']:
                if elbow in pose_kps:
                    pt = tuple(np.array(pose_kps[elbow]).astype(int))
                    cv2.circle(u_mask, pt, int(t_arm * 0.55), 255, -1)
            
            # Mask by total garment (Sleeves & Torso)
            u_mask = cv2.bitwise_and(u_mask, cv2.bitwise_not(garment_mask))
            
            # --- LOWER ARMS & HANDS: Masked ONLY by Sleeves (Allowed on Torso) ---
            # User: "allowed to appear on top of the shirt torso, but will still be hidden behind the sleeves"
            l_mask = np.zeros((H, W), dtype=np.uint8)
            for s, e, t in lower_segments:
                if s in pose_kps and e in pose_kps:
                    self.draw_limb(l_mask, pose_kps[s], pose_kps[e], t)
            
            # Ensure continuity at wrist for hands
            for wrist in ['left_wrist', 'right_wrist']:
                if wrist in pose_kps:
                    pt = tuple(np.array(pose_kps[wrist]).astype(int))
                    cv2.circle(l_mask, pt, int(t_arm * 0.55), 255, -1)
                    
            l_mask = cv2.bitwise_and(l_mask, cv2.bitwise_not(sleeve_mask))
            
            # Merge
            arm_total = cv2.bitwise_or(u_mask, l_mask)
            
            # Final protection for arms from Upper Body Mask (if arms cross coat)
            if upper_body_mask is not None:
                # Ensure upper_body_mask is binary same size
                if upper_body_mask.shape != arm_total.shape[:2]:
                    upper_body_mask = cv2.resize(upper_body_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                arm_total = cv2.bitwise_and(arm_total, cv2.bitwise_not(upper_body_mask))
                
            combined_mask = cv2.bitwise_or(combined_mask, arm_total)
        
        return combined_mask

    def process(self, person_img, garment_mask, output_dir, pose_kps=None, include_arms=True, include_legs=True, sampling_img=None, upper_body_mask=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if person_img is None or garment_mask is None:
            print(f"Error: Missing images")
            return None
        
        if pose_kps is None:
            print("Detecting pose...")
            pose_kps = self.detect_pose(person_img)
            if not pose_kps:
                print("No pose detected!")
                return person_img
        
        # Determine sampling image (Defaults to person_img if not provided)
        # Use full original person image to avoid sampling the applied garment
        s_img = sampling_img if sampling_img is not None else person_img
        
        # Estimate skin color
        skin_color = self.estimate_skin_color(s_img, pose_kps)
        print(f"  🎨 Estimated Skin Color: {skin_color}")
        
        print("Generating artificial skin mask...")
        skin_mask = self.generate_skin_mask(person_img, garment_mask, pose_kps, include_arms=include_arms, include_legs=include_legs, upper_body_mask=upper_body_mask)
        
        # Create the artificial skin image
        skin_img = np.zeros_like(person_img)
        skin_img[:] = skin_color
        skin_img = cv2.bitwise_and(skin_img, skin_img, mask=skin_mask)
        
        # Composite onto person image
        alpha = cv2.GaussianBlur(skin_mask, (7, 7), 0).astype(float) / 255.0
        alpha = alpha[..., None]
        
        result = (skin_img * alpha + person_img * (1 - alpha)).astype(np.uint8)
        
        # Save outputs
        cv2.imwrite(str(output_dir / "artificial_skin_mask.png"), skin_mask)
        cv2.imwrite(str(output_dir / "artificial_skin_result.png"), result)
        
        return result, skin_mask

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Artificial Skin Masking Helper")
    parser.add_argument("--person", type=str, required=True, help="Path to person image")
    parser.add_argument("--mask", type=str, required=True, help="Path to garment mask (FINAL_MASK.png)")
    parser.add_argument("--out", type=str, default="skin_output", help="Output directory")
    
    args = parser.parse_args()
    
    helper = ArtificialSkinHelper()
    helper.process(args.person, args.mask, args.out)
