import cv2
import numpy as np
import os
import sys
from pathlib import Path
from modules.virtual_tryon2.shirt import FixedShirtPantsWarper
from modules.virtual_tryon2.pants import LBSPantsWarper
from modules.virtual_tryon2.universal_vton import UniversalGarmentWarper

class VTONCoordinator:
    def __init__(self, b2_model_path, b3_model_path):
        self.b2_model_path = b2_model_path
        self.b3_model_path = b3_model_path
        
    def detect_garment_type(self, cloth_img):
        """
        Automatically detects garment type using SegFormer-B3.
        Returns: 'upper', 'lower', 'overall', or 'outfit'
        """
        print("\n[Coordinator] Automatically detecting garment type...")
        
        temp_warper = FixedShirtPantsWarper(self.b2_model_path, self.b3_model_path)
        seg_map = temp_warper.segment_cloth_b3(cloth_img)
        unique_classes = np.unique(seg_map).tolist()
        
        # Define class groups (Strict according to B3 legend)
        UPPER_CLASSES = {1, 2, 3, 4, 5, 6, 10, 28, 29, 32, 34, 39, 40, 41, 42, 43, 44, 45, 46}
        LOWER_CLASSES = {7, 8, 9, 33}
        OVERALL_CLASSES = {11, 12, 13} # Dress, Jumpsuit, Cape only (Remove 20 - Belt)
        
        # Count pixels for confidence
        upper_pixels = np.sum(np.isin(seg_map, list(UPPER_CLASSES)))
        lower_pixels = np.sum(np.isin(seg_map, list(LOWER_CLASSES)))
        dress_pixels = np.sum(np.isin(seg_map, list(OVERALL_CLASSES)))

        print(f"  [Detection] Analysis -> Upper Pixels: {upper_pixels}, Lower Pixels: {lower_pixels}, Dress Pixels: {dress_pixels}")

        # RULE 1: If Dress/Overall label is substantial, it's ALWAYS 'overall' (Universal Mode)
        # Even if pants are present, if it's a dress outfit, universal is better.
        if dress_pixels > 5000:
            print("  👗 Dress/Overall detected. Prioritizing Universal mode.")
            return 'overall'
        
        # RULE 2: If both Upper and Lower are significant, it's an 'outfit' (Pants then Shirt)
        if upper_pixels > 5000 and lower_pixels > 5000:
            print("  👔👖 Both Upper and Lower detected. Initiating Full Outfit mode (Pants -> Shirt).")
            return 'outfit'
        
        # RULE 3: Majority rule for single items
        counts = {'upper': upper_pixels, 'lower': lower_pixels, 'overall': dress_pixels}
        detected = max(counts, key=counts.get)
        
        if counts[detected] < 1000:
            print("  ⚠ Low confidence detection. Defaulting to 'overall'.")
            return 'overall'
            
        print(f"  [Detection] Result: {detected}")
        return detected

    def process(self, person_path, cloth_path, output_dir, mode=None):
        person_img = cv2.imread(str(person_path))
        cloth_img = cv2.imread(str(cloth_path))
        
        if person_img is None or cloth_img is None:
            print("✗ Error: Failed to load images.")
            return

        if mode is None:
            print("\n" + "="*50)
            print(" VTON COORDINATOR - MODE SELECTION")
            print("="*50)
            print("1. Upper Clothing (Shirt, Top, Outwear)")
            print("2. Lower Clothing (Pants, Shorts, Skirt)")
            print("3. Overall (Dress, Jumpsuit, or Full-set Universal)")
            print("4. Auto-Detect (Recommend)")
            print("5. Upper and Lower Cloth (Sequential Workflow)")
            print("="*50)
            
            choice = input("Select mode [1-5] or press Enter for Auto: ").strip()
            
            if choice == '1': mode = 'upper'
            elif choice == '2': mode = 'lower'
            elif choice == '3': mode = 'overall'
            elif choice == '5': mode = 'outfit'
            else: mode = 'auto'

        if mode == 'auto':
            mode = self.detect_garment_type(cloth_img)

        print(f"\n[Coordinator] Workflow: {mode.upper()}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        result = None

        if mode == 'lower':
            warper = LBSPantsWarper(self.b2_model_path, self.b3_model_path)
            target_pants_path = output_dir / "final_result.png"
            warper.process(str(person_path), str(cloth_path), str(target_pants_path))
            result = cv2.imread(str(target_pants_path))
            
        elif mode == 'upper':
            warper = FixedShirtPantsWarper(self.b2_model_path, self.b3_model_path)
            result = warper.process(person_img, cloth_img, output_dir)
            
        elif mode == 'outfit':
            # --- TWO-STEP SEQUENTIAL PROCESS ---
            print("\n" + "*"*60)
            print(" COORDINATOR: STARTING COMPOSITE OUTFIT WORKFLOW")
            print("*"*60)
            
            # STEP 1: Apply Pants first
            print("\n[Coordinator] OUTFIT STEP 1/2: Applying Lower Garment (pants.py)...")
            pants_warper = LBSPantsWarper(self.b2_model_path, self.b3_model_path)
            temp_pants_path = output_dir / "temp_pants_step.png"
            pants_warper.process(str(person_path), str(cloth_path), str(temp_pants_path))
            
            # Preserve the Pants Mask before it gets overwritten by the shirt step
            src_mask = output_dir / "final_mask.png"
            pants_mask_path = output_dir / "temp_pants_mask.png"
            if src_mask.exists():
                if pants_mask_path.exists(): pants_mask_path.unlink()
                src_mask.rename(pants_mask_path)
                print("  ✓ Preserved Pants Mask for merging.")

            # Load result of step 1 as input for step 2
            pants_only_image = cv2.imread(str(temp_pants_path))
            if pants_only_image is None:
                print("✗ Outfit Sequential Step 1 failed. Aborting.")
                return None
                
            # STEP 2: Apply Shirt on top of the Pants Result
            print("\n[Coordinator] OUTFIT STEP 2/2: Applying Upper Garment (shirt.py)...")
            print("  🔧 SPECIAL MODE: Combined outfit detected - enforcing shoulder-edge alignment")
            shirt_warper = FixedShirtPantsWarper(self.b2_model_path, self.b3_model_path)
            # Pass flag to indicate this is a combined outfit (both upper + lower)
            result = shirt_warper.process(pants_only_image, cloth_img, output_dir, original_img=pants_only_image, combined_outfit=True)

            # --- REFINED MASK MERGING ---
            # Goal: Shirt should "hide" the pants underneath it.
            print("\n[Coordinator] Merging Garment Masks (Layered)...")
            shirt_mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            pants_mask = cv2.imread(str(pants_mask_path), cv2.IMREAD_GRAYSCALE)

            if shirt_mask is not None and pants_mask is not None:
                # 1. Create a "Solid Silhouette" of the shirt (fill all holes)
                # This represents the total area 'occupied' by the shirt layer.
                silhouette = shirt_mask.copy()
                contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(silhouette, contours, 255)
                
                # 2. "Punch out" the pants mask where the shirt silhouette exists
                # This prevents pants pixels from showing through shirt gaps (like the torso center).
                pants_mask_cleaned = cv2.bitwise_and(pants_mask, cv2.bitwise_not(silhouette))
                
                # 3. Add the actual shirt mask (with its necessary gaps for skin) on top
                combined_mask = cv2.bitwise_or(shirt_mask, pants_mask_cleaned)
                
                cv2.imwrite(str(src_mask), combined_mask)
                print("  ✓ Corrected Layered Mask saved to final_mask.png")
            
            print("\n✅ Composite Outfit Workflow Finished!")
            
        else: # overall / universal
            warper = UniversalGarmentWarper(self.b2_model_path, self.b3_model_path)
            result = warper.process(person_img, cloth_img, output_dir)

        if result is not None:
            final_out = output_dir / "final_result.png"
            cv2.imwrite(str(final_out), result)
            
            # --- USER REQUEST: CLEANUP INTERMEDIATE IMAGES ---
            # Keep only 'final_result.png' and 'final_mask.png'
            print("[Coordinator] Cleaning up intermediate debug images...")
            keep_files = {"final_result.png", "final_mask.png"}
            for file_path in output_dir.glob("*.png"):
                if file_path.name not in keep_files:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"  ⚠ Failed to delete {file_path.name}: {e}")
            
            print(f"\n✅ All process complete. Saved to: {final_out}")
        else:
            print("\n✗ Warp failed.")
        
        return result

def main():
    # Discover project root (4 levels up from Modules/Virtual_Tryon2)
    base_dir = Path(__file__).resolve().parents[3]
    
    # Standardized paths from models_downloader.py
    B2_MODEL = base_dir / "models" / "SegFormerB2Clothes" / "segformer_b2_clothes.onnx"
    B3_MODEL = base_dir / "models" / "segformer-b3-fashion"
    
    # Check if models exist
    if not B2_MODEL.exists():
        print(f"❌ Error: B2 model not found at {B2_MODEL}")
    if not B3_MODEL.exists():
        print(f"❌ Error: B3 model not found at {B3_MODEL}")
    
    # Example paths
    PERSON = r"c:\Users\PC\Downloads\232fa7aa-4854-41d8-9c78-576430912fd81723280009308-FableStreet-LivIn-Bootcut-Trousers-5211723280009216-6.jpg"
    CLOTH = r"c:\Users\PC\Downloads\Gemini_Generated_Image_oh8v93oh8v93oh8v.png"
    OUTPUT = Path(r"C:\Users\PC\.gemini\antigravity\scratch\coordinator_test")
    
    coordinator = VTONCoordinator(B2_MODEL, B3_MODEL)
    coordinator.process(PERSON, CLOTH, OUTPUT)

if __name__ == "__main__":
    main()
