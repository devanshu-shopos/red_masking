#!/usr/bin/env python3
"""
Minimal Setup Script for AI Red Masking
Downloads models and sets up lightweight environment
"""
import os
import sys
import subprocess
from pathlib import Path
import urllib.request

def run_command(cmd, description):
    """Run shell command with description"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def download_file(url, path, description):
    """Download file with progress"""
    print(f"📥 Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ Downloaded {description} ({size_mb:.0f}MB)")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def setup_minimal():
    """Complete minimal setup"""
    print("🚀 MINIMAL AI RED MASKING SETUP")
    print("=" * 40)
    
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 1. Create virtual environment
    venv_dir = base_dir / "venv_minimal"
    if not venv_dir.exists():
        if not run_command(f"python3 -m venv {venv_dir}", "Creating virtual environment"):
            return False
    
    # 2. Install dependencies
    pip_cmd = f"{venv_dir}/bin/pip" if os.name != 'nt' else f"{venv_dir}\\Scripts\\pip"
    if not run_command(f"{pip_cmd} install -r requirements_minimal.txt", "Installing dependencies"):
        return False
    
    # 3. Download models
    models = [
        {
            "name": "GroundingDINO model (694MB)",
            "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
            "path": models_dir / "groundingdino_swint_ogc.pth"
        },
        {
            "name": "GroundingDINO config",
            "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
            "path": models_dir / "GroundingDINO_SwinT_OGC.cfg.py"
        },
        {
            "name": "SAM model (375MB)",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "path": models_dir / "sam_vit_b_01ec64.pth"
        }
    ]
    
    for model in models:
        if not model["path"].exists():
            if not download_file(model["url"], model["path"], model["name"]):
                return False
        else:
            print(f"✅ Found: {model['name']}")
    
    # 4. Make script executable
    script_path = base_dir / "red_mask_minimal"
    if script_path.exists():
        os.chmod(script_path, 0o755)
        print("✅ Made red_mask_minimal executable")
    
    print("\n🎉 MINIMAL SETUP COMPLETE!")
    print("=" * 40)
    print("📊 Installation size: ~1.1GB (vs 2.4GB full ComfyUI)")
    print("⚡ Faster startup and processing")
    print("🔧 No web server or GUI dependencies")
    print("")
    print("🎯 Ready to use:")
    print("  ./red_mask_minimal test.png")
    print("  ./red_mask_minimal my_image.jpg custom_output")
    
    return True

if __name__ == "__main__":
    success = setup_minimal()
    sys.exit(0 if success else 1)