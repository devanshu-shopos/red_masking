#!/usr/bin/env python3
"""
Minimal ComfyUI-based AI Red Masking System
Lightweight version using existing ComfyUI nodes without server overhead
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# Add ComfyUI paths
script_dir = Path(__file__).parent
sys.path.append(str(script_dir / "ComfyUI"))
sys.path.append(str(script_dir / "ComfyUI/custom_nodes/comfyui_segment_anything"))

try:
    from node import SAMModelLoader, GroundingDinoModelLoader, GroundingDinoSAMSegment
    print("✅ ComfyUI nodes loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load ComfyUI nodes: {e}")
    print("Make sure ComfyUI and custom nodes are properly installed")
    sys.exit(1)

class MinimalComfyUIDetector:
    """Minimal detector using existing ComfyUI nodes"""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            self.models_dir = script_dir / "ComfyUI/models"
        else:
            self.models_dir = Path(models_dir)
            
        self.groundingdino_model = None
        self.sam_model = None
        
    def load_models(self):
        """Load GroundingDINO and SAM models using ComfyUI nodes"""
        print("📡 Loading GroundingDINO model...")
        groundingdino_loader = GroundingDinoModelLoader()
        self.groundingdino_model = groundingdino_loader.main(
            model_name="GroundingDINO_SwinT_OGC (694MB)"
        )[0]
        print("✅ GroundingDINO loaded")
        
        print("🎯 Loading SAM model...")
        sam_loader = SAMModelLoader()
        self.sam_model = sam_loader.main(
            model_name="sam_vit_b (375MB)"
        )[0]
        print("✅ SAM loaded")
    
    def detect_and_segment(self, image_tensor, prompt="shirt", threshold=0.3):
        """Combined detection and segmentation using ComfyUI nodes"""
        print(f"🎯 Running AI detection for '{prompt}'...")
        
        grounding_sam_segment = GroundingDinoSAMSegment()
        detection_result = grounding_sam_segment.main(
            prompt=prompt,
            threshold=threshold,
            sam_model=self.sam_model,
            grounding_dino_model=self.groundingdino_model,
            image=image_tensor
        )
        
        detected_image = detection_result[0]  # Image with detections
        detected_mask = detection_result[1]   # Mask tensor
        
        print("✅ Detection and segmentation complete")
        return detected_image, detected_mask
    
    def apply_red_masking(self, original_tensor, mask_tensor):
        """Apply red overlay to detected areas"""
        batch, height, width, channels = original_tensor.shape
        
        # Create red overlay
        red_overlay = torch.zeros_like(original_tensor)
        red_overlay[:, :, :, 0] = 1.0  # Pure red
        red_overlay[:, :, :, 1] = 0.0  # No green
        red_overlay[:, :, :, 2] = 0.0  # No blue
        
        # Expand mask to match image dimensions
        if mask_tensor.dim() == 3:
            mask_expanded = mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # Resize mask if needed
        if mask_expanded.shape[1:3] != (height, width):
            mask_expanded = torch.nn.functional.interpolate(
                mask_expanded.permute(0, 3, 1, 2),
                size=(height, width),
                mode='nearest'
            ).permute(0, 2, 3, 1)
        
        # Apply masking: detected areas = red, everything else = original
        result = original_tensor * (1 - mask_expanded) + red_overlay * mask_expanded
        
        return result
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        img_array = (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to tensor"""
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)
    
    def process_image(self, image_path, output_path, prompt="shirt", threshold=0.3):
        """Complete processing pipeline"""
        print(f"📸 Processing: {image_path}")
        
        # Load and convert image
        pil_image = Image.open(image_path).convert('RGB')
        image_tensor = self.pil_to_tensor(pil_image)
        print(f"✅ Image loaded: {pil_image.size}")
        
        # Run detection and segmentation
        detected_image, mask_tensor = self.detect_and_segment(
            image_tensor, prompt, threshold
        )
        
        # Apply red masking
        print("🔴 Applying red overlay...")
        result_tensor = self.apply_red_masking(image_tensor, mask_tensor)
        
        # Calculate coverage
        coverage = torch.sum(mask_tensor).item() / (mask_tensor.numel()) * 100
        
        # Convert back to PIL and save
        result_pil = self.tensor_to_pil(result_tensor)
        result_pil.save(output_path)
        
        print(f"💾 Saved: {output_path}")
        print(f"📊 Coverage: {coverage:.1f}% of image")

def main():
    parser = argparse.ArgumentParser(description="Minimal ComfyUI AI Red Masking")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("-p", "--prompt", default="shirt", help="Detection prompt")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Detection threshold")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.image)
        output_path = input_path.parent / f"minimal_red_masked_{input_path.stem}.png"
    
    try:
        print("🚀 MINIMAL COMFYUI AI RED MASKING")
        print("=" * 40)
        
        # Initialize detector
        detector = MinimalComfyUIDetector()
        
        # Load models
        detector.load_models()
        
        # Process image
        detector.process_image(args.image, output_path, args.prompt, args.threshold)
        
        print("\n🎉 SUCCESS!")
        print(f"🤖 AI detected '{args.prompt}' objects using ComfyUI nodes")
        print(f"🎯 Applied red masking with {args.threshold} threshold")
        print(f"🔴 Only detected areas turned red, background preserved")
        print(f"⚡ Minimal overhead - no ComfyUI server required!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())