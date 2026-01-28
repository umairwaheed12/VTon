import cv2
import numpy as np
from .model_loader import get_b2_session

class ClothRemover:
    """
    Helper class to remove specific clothing items from a person image 
    and replace them with a natural generated background.
    """
    def __init__(self, seg_model_path):
        self.session = get_b2_session(seg_model_path)
        self.input_name = self.session.get_inputs()[0].name if self.session else None

    def _segment_logits(self, image):
        """Runs SegFormer on the image."""
        h, w = image.shape[:2]
        img = cv2.resize(image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        img = img.transpose(2,0,1)[None].astype(np.float32)
        logits = self.session.run(None, {self.input_name: img})[0][0]
        logits = cv2.resize(logits.transpose(1,2,0), (w,h))
        return logits

    def generate_natural_background(self, image):
        """
        Generates a natural background by interpolating from image edges.
        """
        ph, pw = image.shape[:2]
        edge_size = min(80, pw // 6, ph // 6)
        
        background_map = np.zeros_like(image, dtype=np.float32)
        weight_map = np.zeros((ph, pw), dtype=np.float32)
        
        regions = [
            (image[0:edge_size, :], 'top'),
            (image[ph-edge_size:ph, :], 'bottom'),
            (image[:, 0:edge_size], 'left'),
            (image[:, pw-edge_size:pw], 'right')
        ]
        
        for region, side in regions:
            if region.size > 0:
                if side == 'top':
                    for i in range(ph):
                        blend_factor = np.exp(-i / (ph * 0.3))
                        if i < edge_size:
                            background_map[i, :] += region[i, :] * blend_factor
                        else:
                            background_map[i, :] += region[-1, :] * blend_factor
                        weight_map[i, :] += blend_factor
                elif side == 'bottom':
                    for i in range(ph):
                        blend_factor = np.exp(-(ph - 1 - i) / (ph * 0.3))
                        idx = i - (ph - edge_size)
                        if idx >= 0:
                            background_map[i, :] += region[idx, :] * blend_factor
                        else:
                            background_map[i, :] += region[0, :] * blend_factor
                        weight_map[i, :] += blend_factor
                elif side == 'left':
                    for j in range(pw):
                        blend_factor = np.exp(-j / (pw * 0.3))
                        if j < edge_size:
                            background_map[:, j] += region[:, j] * blend_factor
                        else:
                            background_map[:, j] += region[:, -1] * blend_factor
                        weight_map[:, j] += blend_factor
                elif side == 'right':
                    for j in range(pw):
                        blend_factor = np.exp(-(pw - 1 - j) / (pw * 0.3))
                        idx = j - (pw - edge_size)
                        if idx >= 0:
                            background_map[:, j] += region[:, idx] * blend_factor
                        else:
                            background_map[:, j] += region[:, 0] * blend_factor
                        weight_map[:, j] += blend_factor
                        
        weight_map = np.maximum(weight_map, 1e-6)
        background_map = (background_map / weight_map[:, :, np.newaxis]).astype(np.uint8)
        return background_map

    def remove_class_mask(self, image, target_classes, protect_classes=None):
        """Generic removal of specific segmentation classes."""
        logits = self._segment_logits(image)
        seg_map = np.argmax(logits, axis=2).astype(np.uint8)
        
        mask = np.zeros(seg_map.shape, dtype=np.uint8)
        
        # Add Targets
        for lab in target_classes:
            mask[seg_map == lab] = 255
            
        # Remove Protected
        if protect_classes:
            for lab in protect_classes:
                mask[seg_map == lab] = 0
                
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if cv2.countNonZero(mask) == 0:
            return image, False
            
        # Dilate slightly for inpainting coverage
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Inpaint
        result = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        return result, True

    def remove_pants(self, image):
        """Removes Pants (6) and Skirt (5). Preserves Legs/Feet/Shoes."""
        # Target: Pants(6), Skirt(5)
        # Protect: Feet(9, 10), Shoes(24), Legs(12, 13), Skin(11), Upper(4), Dress(7)?
        # If user has a dress, 'pants' removal might cut the bottom of the dress. 
        # But 'pants.py' is for pants.
        print("ACTION: Removing Pants/Skirt...")
        return self.remove_class_mask(image, target_classes={5, 6}, protect_classes={9, 10, 24, 12, 13})

    def remove_shirt(self, image):
        """Removes Upper Clothes (4) and Outer (3). Preserves Arms/Hands/Head."""
        # Target: Upper(4), Outer(3)? (Coat is 3 or 5 or 7 depending on model, usually 4 or 3)
        # Segformer B2 Clothes labels:
        # 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe, 11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf
        # So Upper is 4. Dress is 7.
        # Check if user wears dress(7) - we should probably remove it if replacing shirt.
        print("ACTION: Removing Shirt (Upper Clothes)...")
        # Removing 4 and 7 (Dress usually covers top too).
        # Protecting Arms (14, 15), Hands (not in standard B2? Arms cover it usually), Face(11), Hair(2).
        return self.remove_class_mask(image, target_classes={4, 7}, protect_classes={14, 15, 11, 2, 6, 5})

    def remove_all(self, image):
        """Removes All Clothes (Upper, Lower, Dress). Preserves Head, Hands, Feet."""
        # Target: Upper(4), Skirt(5), Pants(6), Dress(7), Belt(8), Scarf(17), Bag(16)?
        print("ACTION: Removing ALL Clothes...")
        return self.remove_class_mask(image, 
                                      target_classes={4, 5, 6, 7, 8, 17}, 
                                      protect_classes={11, 2, 9, 10, 24, 1, 3}) # Protect Head(11,2,1,3) + Feet(9,10,24)

