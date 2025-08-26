#!/usr/bin/env python3
"""
Simple installation verification for AI Red Masking Workflow
"""
import os
import sys
from pathlib import Path

def main():
    print("🔍 AI RED MASKING INSTALLATION VERIFICATION")
    print("=" * 50)
    
    base_path = Path(__file__).parent
    all_good = True
    
    # Check essential files
    essential_files = [
        "red_mask",
        "process_direct_ai_detection.py", 
        "requirements.txt",
        "README.md",
        "ComfyUI/main.py",
        "comfyui_env/bin/activate" if os.name != 'nt' else "comfyui_env/Scripts/activate.bat"
    ]
    
    print("📁 Checking essential files...")
    for file in essential_files:
        file_path = base_path / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_good = False
    
    # Check AI models
    print("\n🤖 Checking AI models...")
    model_files = [
        "ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth",
        "ComfyUI/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py", 
        "ComfyUI/models/sams/sam_vit_b_01ec64.pth"
    ]
    
    total_size = 0
    for model in model_files:
        model_path = base_path / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            total_size += size_mb
            print(f"  ✅ {model} ({size_mb:.0f}MB)")
        else:
            print(f"  ❌ {model} - MISSING!")
            all_good = False
    
    print(f"\n📊 Total model size: {total_size:.0f}MB")
    
    # Check custom nodes
    print("\n🔧 Checking custom nodes...")
    custom_nodes = [
        "ComfyUI/custom_nodes/comfyui_segment_anything",
        "ComfyUI/custom_nodes/ComfyUI-segment-anything-2",
        "ComfyUI/custom_nodes/ComfyUI_LayerStyle",
        "ComfyUI/custom_nodes/ComfyUI-Impact-Pack"
    ]
    
    for node in custom_nodes:
        node_path = base_path / node
        if node_path.exists() and node_path.is_dir():
            print(f"  ✅ {node.split('/')[-1]}")
        else:
            print(f"  ❌ {node.split('/')[-1]} - MISSING!")
            all_good = False
    
    # Check directories
    print("\n📂 Checking directories...")
    directories = [
        "ComfyUI/input",
        "ComfyUI/output", 
        "comfyui_env"
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ {directory} - MISSING!")
            all_good = False
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 INSTALLATION VERIFIED SUCCESSFULLY!")
        print("   Ready to run: ./red_mask ComfyUI/input/sample_shirt.jpg test")
        print("   Your AI detection system is ready to use!")
    else:
        print("❌ INSTALLATION INCOMPLETE!")
        print("   Please follow the README.md setup instructions.")
        print("   Missing components detected above.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())