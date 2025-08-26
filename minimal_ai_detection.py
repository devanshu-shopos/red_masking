#!/usr/bin/env python3
"""
Minimal Standalone AI Red Masking System
Uses GroundingDINO + SAM for shirt detection without ComfyUI framework
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# Core model imports
# Try to use existing ComfyUI GroundingDINO + SAM nodes first (fallback)
USE_COMFYUI_NODES = False
try:
    # Add ComfyUI paths if available
    script_dir = Path(__file__).parent
    comfyui_path = script_dir / "ComfyUI"
    if comfyui_path.exists():
        sys.path.append(str(comfyui_path))
        sys.path.append(str(comfyui_path / "custom_nodes/comfyui_segment_anything"))
        
        from node import SAMModelLoader, GroundingDinoModelLoader, GroundingDinoSAMSegment
        USE_COMFYUI_NODES = True
        print("✅ Using existing ComfyUI nodes")
except ImportError:
    USE_COMFYUI_NODES = False

if not USE_COMFYUI_NODES:
    try:
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        import groundingdino.datasets.transforms as T
        from segment_anything import sam_model_registry, SamPredictor
        print("✅ Using standalone GroundingDINO + SAM")
    except ImportError as e:
        print(f"❌ Missing dependencies for standalone mode.")
        print(f"For ComfyUI integration: Use existing './red_mask' script") 
        print(f"For standalone: Install dependencies with pip install groundingdino-py segment_anything")
        print(f"Error: {e}")
        sys.exit(1)

class MinimalAIDetector:
    """Lightweight AI detection system using GroundingDINO + SAM"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.device = self._get_device()
        self.groundingdino_model = None
        self.sam_predictor = None
        
        print(f"🔧 Device: {self.device}")
        
    def _get_device(self):
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps") 
        else:
            return torch.device("cpu")
    
    def load_groundingdino(self, model_path=None, config_path=None):
        """Load GroundingDINO model"""
        if model_path is None:
            model_path = self.models_dir / "groundingdino_swint_ogc.pth"
        if config_path is None:
            config_path = self.models_dir / "GroundingDINO_SwinT_OGC.cfg.py"
            
        if not model_path.exists():
            raise FileNotFoundError(f"GroundingDINO model not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
            
        print("📡 Loading GroundingDINO model...")
        
        # Load config and model
        args = SLConfig.fromfile(str(config_path))
        args.device = str(self.device)
        model = build_model(args)
        
        # Load weights
        checkpoint = torch.load(str(model_path), map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"✅ GroundingDINO loaded: {load_res}")
        
        model.eval()
        model = model.to(self.device)
        self.groundingdino_model = model
        
    def load_sam(self, model_path=None):
        """Load SAM model"""
        if model_path is None:
            model_path = self.models_dir / "sam_vit_b_01ec64.pth"
            
        if not model_path.exists():
            raise FileNotFoundError(f"SAM model not found: {model_path}")
            
        print("🎯 Loading SAM model...")
        sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("✅ SAM loaded successfully")
        
    def detect_objects(self, image_pil, text_prompt="shirt", threshold=0.3):
        """Detect objects using GroundingDINO"""
        if self.groundingdino_model is None:
            raise RuntimeError("GroundingDINO model not loaded")
            
        # Prepare image transforms
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Transform image
        image_transformed, _ = transform(image_pil, None)
        
        # Run detection
        with torch.no_grad():
            outputs = self.groundingdino_model(image_transformed[None].to(self.device), captions=[text_prompt])
            
        # Process results
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
        
        # Filter by threshold
        mask = prediction_logits.max(dim=1)[0] > threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
        
        # Convert to absolute coordinates
        h, w = image_pil.size[1], image_pil.size[0]
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        
        return boxes, logits
    
    def segment_objects(self, image_pil, boxes):
        """Create segmentation masks using SAM"""
        if self.sam_predictor is None:
            raise RuntimeError("SAM model not loaded")
            
        # Convert PIL to numpy
        image_np = np.array(image_pil)
        self.sam_predictor.set_image(image_np)
        
        # Generate masks for all boxes
        masks = []
        for box in boxes:
            # Convert box format for SAM
            box_xyxy = box.numpy()
            
            # Generate mask
            mask, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy[None, :],
                multimask_output=False,
            )
            masks.append(mask[0])
            
        return masks
    
    def apply_red_masking(self, image_pil, masks):
        """Apply red overlay to detected areas"""
        if not masks:
            print("⚠️ No masks found, returning original image")
            return image_pil
            
        # Convert to numpy
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        
        # Combine all masks
        combined_mask = np.zeros((h, w), dtype=bool)
        for mask in masks:
            if mask.shape != (h, w):
                # Resize mask if needed
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((w, h), PILImage.Resampling.NEAREST)
                mask = np.array(mask_pil) > 128
            combined_mask |= mask
        
        # Create red overlay
        red_overlay = image_np.copy()
        red_overlay[combined_mask, 0] = 255  # Pure red
        red_overlay[combined_mask, 1] = 0    # No green
        red_overlay[combined_mask, 2] = 0    # No blue
        
        # Apply masking: detected areas = red, everything else = original
        result = np.where(combined_mask[..., None], red_overlay, image_np)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def process_image(self, image_path, output_path, prompt="shirt", threshold=0.3):
        """Complete pipeline: detect -> segment -> apply red masking"""
        print(f"📸 Processing: {image_path}")
        
        # Load image
        image_pil = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded: {image_pil.size}")
        
        # Detect objects
        print(f"🎯 Detecting '{prompt}' objects...")
        boxes, logits = self.detect_objects(image_pil, prompt, threshold)
        print(f"✅ Found {len(boxes)} detections")
        
        if len(boxes) == 0:
            print("⚠️ No objects detected, saving original image")
            image_pil.save(output_path)
            return
        
        # Segment objects
        print("🔍 Creating segmentation masks...")
        masks = self.segment_objects(image_pil, boxes)
        print(f"✅ Generated {len(masks)} masks")
        
        # Apply red masking
        print("🔴 Applying red overlay...")
        result_image = self.apply_red_masking(image_pil, masks)
        
        # Calculate coverage
        total_pixels = image_pil.size[0] * image_pil.size[1]
        masked_pixels = sum(np.sum(mask) for mask in masks)
        coverage = (masked_pixels / total_pixels) * 100
        
        # Save result
        result_image.save(output_path)
        print(f"💾 Saved: {output_path}")
        print(f"📊 Coverage: {coverage:.1f}% of image")

def download_models(models_dir="models"):
    """Download required models if not present"""
    import urllib.request
    
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    
    models = [
        {
            "name": "GroundingDINO model",
            "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
            "path": models_dir / "groundingdino_swint_ogc.pth",
            "size": "694MB"
        },
        {
            "name": "GroundingDINO config",
            "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
            "path": models_dir / "GroundingDINO_SwinT_OGC.cfg.py", 
            "size": "1KB"
        },
        {
            "name": "SAM model",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "path": models_dir / "sam_vit_b_01ec64.pth",
            "size": "375MB"
        }
    ]
    
    for model in models:
        if not model["path"].exists():
            print(f"📥 Downloading {model['name']} ({model['size']})...")
            try:
                urllib.request.urlretrieve(model["url"], model["path"])
                print(f"✅ Downloaded: {model['path']}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return False
        else:
            print(f"✅ Found: {model['name']}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Minimal AI Red Masking")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("-p", "--prompt", default="shirt", help="Detection prompt (default: shirt)")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Detection threshold (default: 0.3)")
    parser.add_argument("--download-models", action="store_true", help="Download required models")
    parser.add_argument("--models-dir", default="models", help="Models directory (default: models)")
    
    args = parser.parse_args()
    
    if args.download_models:
        print("🔽 Downloading models...")
        if download_models(args.models_dir):
            print("✅ All models downloaded successfully!")
        else:
            print("❌ Model download failed!")
            return 1
        
        if not args.image:
            return 0
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.image)
        output_path = input_path.parent / f"red_masked_{input_path.stem}.png"
    
    try:
        # Initialize detector
        detector = MinimalAIDetector(args.models_dir)
        
        # Load models
        detector.load_groundingdino()
        detector.load_sam()
        
        # Process image
        detector.process_image(args.image, output_path, args.prompt, args.threshold)
        
        print("\n🎉 SUCCESS!")
        print(f"🤖 AI detected '{args.prompt}' objects")
        print(f"🎯 Applied red masking with {args.threshold} threshold") 
        print(f"🔴 Only detected areas turned red, background preserved")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())