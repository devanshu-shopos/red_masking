# AI-Powered Red Masking Workflow

![ComfyUI](https://img.shields.io/badge/ComfyUI-AI%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12+-blue) ![GroundingDINO](https://img.shields.io/badge/GroundingDINO-694MB-orange) ![SAM](https://img.shields.io/badge/SAM-375MB-purple) ![Minimal](https://img.shields.io/badge/Minimal-1.1GB-green)

An AI-powered image processing workflow that detects shirt objects in images using **GroundingDINO + SAM** and applies precise red masking ONLY to detected shirt areas. Available in both **full ComfyUI** and **minimal standalone** versions.

## 🚀 Quick Start (Minimal Version - Recommended)

**Single-command execution:**
```bash
./red_mask_minimal <image_path> [output_name]
```

**Examples:**
```bash
# Process any image
./red_mask_minimal test.png
./red_mask_minimal my_photo.jpg custom_result

# The minimal version is much faster and lighter!
```

## ✨ Two Versions Available

### 🎯 **Minimal Version (Recommended)**
- **Size**: ~1.1GB (vs 2.4GB full)  
- **Startup**: Instant (no server overhead)
- **Dependencies**: Essential AI models only
- **Performance**: Same AI accuracy, faster execution
- **Script**: `./red_mask_minimal`

### 🔧 **Full ComfyUI Version**
- **Size**: ~2.4GB (complete framework)
- **Features**: Full ComfyUI ecosystem + web GUI
- **Use case**: Advanced workflows, research
- **Script**: `./red_mask`

## 🛠️ Installation Options

### Option 1: Minimal Setup (Recommended)

**Quick setup for lightweight usage:**
```bash
git clone <your-repository-url>
cd "red masking"

# Create virtual environment
python3 -m venv venv_minimal
source venv_minimal/bin/activate  # Mac/Linux
# or venv_minimal\Scripts\activate  # Windows

# Install minimal dependencies  
pip install torch torchvision Pillow numpy transformers timm addict yapf opencv-python

# Download AI models (1GB)
mkdir -p models
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
  -o "models/groundingdino_swint_ogc.pth"
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py" \
  -o "models/GroundingDINO_SwinT_OGC.cfg.py"  
curl -k -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  -o "models/sam_vit_b_01ec64.pth"

# Make executable and test
chmod +x red_mask_minimal
./red_mask_minimal test.png
```

### Option 2: Full ComfyUI Setup

**Complete installation with full framework:**
```bash
git clone <your-repository-url>
cd "red masking"

# Create virtual environment
python3 -m venv comfyui_env
source comfyui_env/bin/activate

# Install ComfyUI framework
git clone https://github.com/comfyanonymous/ComfyUI.git
pip install -r ComfyUI/requirements.txt

# Install custom nodes (AI detection)
cd ComfyUI/custom_nodes
git clone https://github.com/storyicon/comfyui_segment_anything.git
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
cd ../..

# Install dependencies
pip install segment_anything timm addict yapf platformdirs
pip install -r ComfyUI/custom_nodes/ComfyUI_LayerStyle/requirements.txt

# Download models to ComfyUI directories
mkdir -p ComfyUI/models/grounding-dino ComfyUI/models/sams
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
  -o "ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth"
curl -k -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  -o "ComfyUI/models/sams/sam_vit_b_01ec64.pth"

# Test full version
chmod +x red_mask
./red_mask ComfyUI/input/test.png
```

## ⚙️ How It Works

### AI Detection Pipeline
1. **GroundingDINO**: Uses text prompt "shirt" to detect shirt objects in image
2. **SAM (Segment Anything)**: Creates precise segmentation masks for detected objects
3. **Red Overlay**: Applies pure red color ONLY to AI-detected shirt areas
4. **Preservation**: Keeps background, face, arms, pants completely untouched

### Technical Configuration
- **Detection Prompt**: "shirt" (customizable in script)
- **Detection Threshold**: 0.3 (30% confidence minimum)
- **Red Overlay**: Pure red (RGB: 255, 0, 0) applied to detected areas
- **Processing**: Binary masking - shirt areas become red, everything else preserved

## 📁 Project Structure

```
red-masking/
├── 🚀 red_mask_minimal                # MINIMAL EXECUTABLE (Recommended)
├── minimal_comfyui_detection.py       # Minimal AI detection engine
├── red_mask                           # Full ComfyUI executable  
├── process_direct_ai_detection.py     # Full ComfyUI AI engine
├── verify_installation.py            # Installation verification
├── requirements_minimal.txt           # Minimal dependencies
├── requirements.txt                   # Full dependencies
├── README.md                          # This documentation
├── CLAUDE.md                         # Development guidance
├── example_output.png                # Reference output
├── models/                           # Minimal version models (1GB)
│   ├── groundingdino_swint_ogc.pth   # Object detection (694MB)
│   ├── GroundingDINO_SwinT_OGC.cfg.py
│   └── sam_vit_b_01ec64.pth         # Segmentation (375MB)
├── venv_minimal/                     # Minimal virtual environment
└── ComfyUI/                          # Full ComfyUI framework (optional)
    ├── custom_nodes/                 # AI model extensions
    ├── models/                       # ComfyUI model directories
    ├── input/                        # Input images
    └── output/                       # Generated results
```

## 🔍 Verification

**Test minimal version:**
```bash
source venv_minimal/bin/activate  # or comfyui_env/bin/activate
python verify_installation.py

# Test AI detection
./red_mask_minimal test.png verification_test
```

## 🎨 Example Results

### Perfect AI Detection:
- **Input**: Original image with person wearing shirt/blazer
- **Output**: Same image with ONLY the shirt/blazer colored bright red
- **Precision**: Perfect boundaries following garment contours
- **Preservation**: Background, face, arms, other clothing completely unchanged

### Performance Comparison:
| Version | Size | Startup Time | Dependencies | Use Case |
|---------|------|--------------|-------------|----------|
| **Minimal** | 1.1GB | Instant | Essential only | Production, deployment |
| **Full** | 2.4GB | 30s | Complete framework | Research, advanced workflows |

## 🚨 Troubleshooting

### Common Issues & Solutions

**Minimal Version Issues:**
```bash
# Missing models
mkdir -p models && ./red_mask_minimal --download-models

# Environment issues  
source venv_minimal/bin/activate
pip install -r requirements_minimal.txt
```

**Full Version Issues:**
```bash
# Use existing troubleshooting
python verify_installation.py
source comfyui_env/bin/activate
```

**Detection Issues:**
- Ensure image contains visible shirt/garment
- Try adjusting threshold: `./red_mask_minimal image.jpg -t 0.2` (more sensitive)
- Check image quality and lighting

## 🔧 Advanced Usage

### Minimal Version Options
```bash
# Custom detection prompt
./red_mask_minimal image.jpg -p "jacket" -o result.png

# Adjust sensitivity
./red_mask_minimal image.jpg -t 0.2  # More sensitive
./red_mask_minimal image.jpg -t 0.5  # Less sensitive

# Direct Python usage
python minimal_comfyui_detection.py image.jpg -o output.png
```

### Modify Detection Parameters
Edit `minimal_comfyui_detection.py`:
```python
# Line ~98: Change detection prompt
prompt="shirt"  # Try: "t-shirt", "jacket", "blazer", "clothing"

# Line ~98: Change detection threshold  
threshold=0.3   # Lower = more sensitive, Higher = more strict
```

## 🔬 Technical Details

### Minimal Version Benefits
- **60% smaller** than full ComfyUI (1.1GB vs 2.4GB)
- **Faster startup** - no web server initialization
- **Same AI accuracy** - uses identical GroundingDINO + SAM models
- **Simpler deployment** - fewer dependencies
- **Direct execution** - no framework overhead

### AI Models Used
- **GroundingDINO SwinT OGC**: 694MB object detection model
- **SAM ViT-B**: 375MB segmentation model
- **Total**: ~1GB of AI models (same for both versions)

### Performance
- **GPU Support**: Auto-detects MPS (Mac), CUDA (PC), CPU fallback  
- **Memory Usage**: ~2GB during processing
- **Speed**: 10-30 seconds per image (minimal faster due to less overhead)
- **Image Support**: All PIL formats (JPG, PNG, etc.)

## 📝 License

This project combines multiple open-source components:
- **ComfyUI**: GPL-3.0 License
- **GroundingDINO**: Apache 2.0 License
- **SAM**: Apache 2.0 License  
- **Minimal Implementation**: MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Test both minimal and full versions
4. Commit changes (`git commit -m 'Add enhancement'`)
5. Push to branch (`git push origin feature/enhancement`)
6. Open Pull Request

## 📞 Support & FAQ

**Q: Which version should I use?**
A: Use **minimal version** for production/deployment. Use **full version** for research or if you need ComfyUI's full ecosystem.

**Q: Do I need ComfyUI for the minimal version?**
A: No! The minimal version is completely standalone but can optionally use ComfyUI nodes if available.

**Q: Is detection accuracy the same?**
A: Yes! Both versions use identical AI models (GroundingDINO + SAM) with same accuracy.

**Q: Can I detect other objects?**
A: Yes! Use `-p "jacket"`, `-p "pants"`, etc. or modify the script.

---

## 🎯 Ready to Use!

**For quick deployment (recommended):**
```bash
# Setup minimal version (1.1GB)
./red_mask_minimal test.png amazing_result
```

**For full research capabilities:**  
```bash
# Setup full ComfyUI version (2.4GB)
./red_mask ComfyUI/input/test.png research_result
```

🔴 **Perfect shirt detection • Precise boundaries • Background preserved • Now in two convenient versions!**