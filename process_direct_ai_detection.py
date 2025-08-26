#!/usr/bin/env python3
"""
Direct AI-based shirt detection using GroundingDINO + SAM without full ComfyUI initialization
"""
import os
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
import numpy as np
from PIL import Image

# Store arguments before imports  
original_argv = sys.argv.copy()
image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_shirt.jpg"
output_prefix = sys.argv[2] if len(sys.argv) > 2 else "direct_ai_detection"

# Clear argv to avoid conflicts
sys.argv = [sys.argv[0]]

print("🤖 DIRECT AI SHIRT DETECTION (GroundingDINO + SAM)")
print("=" * 55)
print(f"Input: {image_path}")
print(f"Output: {output_prefix}")

# Add paths
sys.path.append('/Users/devanshurana/Desktop/ShopOS/red masking/ComfyUI')
sys.path.append('/Users/devanshurana/Desktop/ShopOS/red masking/ComfyUI/custom_nodes/comfyui_segment_anything')

# Import the nodes directly
from node import SAMModelLoader, GroundingDinoModelLoader, GroundingDinoSAMSegment

def prepare_input_image(image_path: str) -> torch.Tensor:
    """Load and prepare input image as tensor"""
    script_dir = Path(__file__).parent
    
    # Try different paths for the image
    if Path(image_path).exists():
        full_path = Path(image_path)
    elif (script_dir / image_path).exists():
        full_path = script_dir / image_path
    else:
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    # Load image using PIL and convert to tensor format expected by ComfyUI
    pil_image = Image.open(full_path).convert('RGB')
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    
    # Convert to tensor in format [batch, height, width, channels]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
    
    print(f"✅ Image loaded: {img_tensor.shape}")
    return img_tensor

def save_result(image_tensor: torch.Tensor, filename_prefix: str):
    """Save result image"""
    # Convert tensor back to PIL image
    img_array = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_array)
    
    # Save to output directory
    output_dir = Path(__file__).parent / "ComfyUI/output"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{filename_prefix}_00001_.png"
    pil_image.save(output_path)
    print(f"📁 Output saved: {output_path.resolve()}")

def main():
    print("🔄 Loading AI models...")
    
    with torch.inference_mode():
        # Load GroundingDINO model
        print("📡 Loading GroundingDINO model...")
        try:
            groundingdino_loader = GroundingDinoModelLoader()
            groundingdino_model = groundingdino_loader.main(
                model_name="GroundingDINO_SwinT_OGC (694MB)"
            )
            print("✅ GroundingDINO model loaded successfully")
        except Exception as e:
            print(f"❌ GroundingDINO loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Load SAM model
        print("🎯 Loading SAM model...")
        try:
            sam_loader = SAMModelLoader()
            sam_model = sam_loader.main(model_name="sam_vit_b (375MB)")
            print("✅ SAM model loaded successfully")
        except Exception as e:
            print(f"❌ SAM loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Load input image
        print(f"📸 Loading image: {image_path}")
        image_tensor = prepare_input_image(image_path)

        # AI-BASED SHIRT DETECTION using GroundingDINO + SAM
        print("🎯 Running AI shirt detection with GroundingDINO + SAM...")
        try:
            grounding_sam_segment = GroundingDinoSAMSegment()
            
            # This is the KEY step - AI detects "shirt" objects in the image
            detection_result = grounding_sam_segment.main(
                prompt="shirt",  # Text prompt for detection
                threshold=0.3,   # Detection confidence threshold
                sam_model=sam_model[0],
                grounding_dino_model=groundingdino_model[0],
                image=image_tensor,
            )
            print("✅ AI shirt detection completed - shirt objects found!")
            
            detected_image = detection_result[0]  # Image with detected areas
            detected_mask = detection_result[1]   # Mask of detected areas
            
            print(f"🔍 Detection results: image={detected_image.shape}, mask={detected_mask.shape}")
            
        except Exception as e:
            print(f"❌ AI detection failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Create red overlay for detected areas
        print("🔴 Creating red overlay for AI-detected shirt areas...")
        
        # Use original image dimensions
        batch, height, width, channels = image_tensor.shape
        red_overlay = torch.zeros_like(image_tensor)
        red_overlay[:, :, :, 0] = 1.0  # Pure red channel
        red_overlay[:, :, :, 1] = 0.0  # No green
        red_overlay[:, :, :, 2] = 0.0  # No blue
        
        # Process mask - ensure it matches image dimensions
        if detected_mask.dim() == 3:
            mask_expanded = detected_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            mask_expanded = detected_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
            
        # Resize mask if needed
        if mask_expanded.shape[1:3] != (height, width):
            mask_expanded = torch.nn.functional.interpolate(
                mask_expanded.permute(0, 3, 1, 2), 
                size=(height, width), 
                mode='nearest'
            ).permute(0, 2, 3, 1)
        
        # Blend: original image where mask is 0, red overlay where mask is 1
        final_result = image_tensor * (1 - mask_expanded) + red_overlay * mask_expanded
        
        # Calculate coverage statistics
        mask_coverage = torch.sum(mask_expanded[:,:,:,0]).item() / (height * width) * 100
        
        print(f"✅ Red overlay applied to {mask_coverage:.1f}% of image")
        
        # Save the final result
        print("💾 Saving AI-detected shirt result...")
        save_result(final_result, output_prefix)

        print("\n🎉 DIRECT AI SHIRT DETECTION SUCCESS!")
        print("🤖 GroundingDINO detected shirt objects in the image")
        print("🎯 SAM created precise segmentation masks")
        print("🔴 Red overlay applied ONLY to AI-detected shirt areas")
        print(f"📊 Coverage: {mask_coverage:.1f}% of image area")

if __name__ == "__main__":
    main()