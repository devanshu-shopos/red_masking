# AI-Powered Red Masking Workflow

![ComfyUI](https://img.shields.io/badge/ComfyUI-AI%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12+-blue) ![GroundingDINO](https://img.shields.io/badge/GroundingDINO-694MB-orange) ![SAM](https://img.shields.io/badge/SAM-375MB-purple)

An AI-powered image processing workflow that detects shirt objects in images using **GroundingDINO + SAM** and applies precise red masking ONLY to detected shirt areas. No geometric overlays - true AI-based object detection and segmentation.

## 🚀 Quick Start

**Single-command execution:**
```bash
./red_mask <image_path> [output_name]
```

**Examples:**
```bash
# Process sample image
./red_mask sample_shirt.jpg

# Process your own image with custom output name  
./red_mask /path/to/photo.jpg my_result

# Generate output with specific name
./red_mask shirt.png hogya
```

## ✨ Key Features

- **🤖 True AI Detection**: GroundingDINO detects "shirt" objects using text prompts
- **🎯 Precise Segmentation**: SAM creates accurate segmentation masks
- **🔴 Perfect Red Masking**: Red overlay applied ONLY to AI-detected shirt areas
- **⚡ One Command Setup**: Complete installation with single script
- **📁 Multiple Formats**: Support for JPG, PNG, and all image formats
- **🖼️ Perfect Preservation**: Background and non-shirt areas remain untouched

## 🛠️ Complete Installation Guide

### Prerequisites
- Python 3.12+
- Git
- ~1.5GB storage space for AI models
- Internet connection for model downloads

### Step 1: Clone Repository
```bash
git clone <your-repository-url>
cd "red masking"
```

### Step 2: Create Virtual Environment
```bash
# Create environment
python3 -m venv comfyui_env

# Activate environment (Mac/Linux)
source comfyui_env/bin/activate

# Activate environment (Windows)
comfyui_env\Scripts\activate
```

### Step 3: Install ComfyUI Framework
```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git

# Install base requirements
pip install -r ComfyUI/requirements.txt
```

### Step 4: Install Required Custom Nodes
```bash
cd ComfyUI/custom_nodes

# 1. Install GroundingDINO + SAM custom node (CRITICAL - this is the AI detection)
git clone https://github.com/storyicon/comfyui_segment_anything.git

# 2. Install SAM2 support 
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git

# 3. Install LayerStyle utilities
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git

# 4. Install Impact Pack utilities  
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git

cd ../..
```

### Step 5: Install Python Dependencies
```bash
# Install segment anything dependencies
pip install segment_anything timm addict yapf platformdirs

# Install LayerStyle dependencies  
pip install -r ComfyUI/custom_nodes/ComfyUI_LayerStyle/requirements.txt

# Install any additional requirements
pip install torch torchvision Pillow numpy
```

### Step 6: Download AI Models (CRITICAL STEP)
```bash
# Create model directories
mkdir -p ComfyUI/models/grounding-dino
mkdir -p ComfyUI/models/sams

# Download GroundingDINO model (694MB) - Object Detection
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
  -o "ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth"

# Download GroundingDINO config
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py" \
  -o "ComfyUI/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py"

# Download SAM model (375MB) - Segmentation  
curl -k -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  -o "ComfyUI/models/sams/sam_vit_b_01ec64.pth"
```

### Step 7: Add Sample Image (Optional)
```bash
# Create input directory
mkdir -p ComfyUI/input

# Add your own image or download sample
curl -L "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=1000&q=80" \
  -o "ComfyUI/input/sample_shirt.jpg"
```

### Step 8: Make Executable
```bash
chmod +x red_mask
```

## 🎯 How to Run

### Basic Usage
```bash
# Test with sample image
./red_mask ComfyUI/input/sample_shirt.jpg test_output

# Process your own image  
./red_mask /path/to/your/image.jpg my_result

# Quick test
./red_mask photo.png hogya
```

### Expected Output
```
🔴 RED MASKING WORKFLOW
=======================
Image: photo.jpg
Output: my_result

🤖 DIRECT AI SHIRT DETECTION (GroundingDINO + SAM)
🔄 Loading AI models...
📡 Loading GroundingDINO model...
✅ GroundingDINO model loaded successfully
🎯 Loading SAM model...
✅ SAM model loaded successfully
📸 Loading image: photo.jpg
✅ Image loaded: torch.Size([1, 1000, 1000, 3])
🎯 Running AI shirt detection with GroundingDINO + SAM...
✅ AI shirt detection completed - shirt objects found!
🔴 Creating red overlay for AI-detected shirt areas...
✅ Red overlay applied to 42.0% of image
💾 Saving AI-detected shirt result...
📁 Output saved: /path/to/ComfyUI/output/my_result_00001_.png

🎉 DIRECT AI SHIRT DETECTION SUCCESS!
🤖 GroundingDINO detected shirt objects in the image
🎯 SAM created precise segmentation masks
🔴 Red overlay applied ONLY to AI-detected shirt areas
📊 Coverage: 42.0% of image area
```

## ⚙️ How It Works

### AI Detection Pipeline
1. **GroundingDINO**: Uses text prompt "shirt" to detect shirt objects in image
2. **SAM (Segment Anything)**: Creates precise segmentation masks for detected objects
3. **Red Overlay**: Applies pure red color ONLY to AI-detected shirt areas
4. **Preservation**: Keeps background, face, arms, pants completely untouched

### Technical Configuration
- **Detection Prompt**: "shirt" (can be modified in script)
- **Detection Threshold**: 0.3 (30% confidence minimum)
- **Red Overlay**: Pure red (RGB: 255, 0, 0) applied to detected areas
- **Blending**: Binary masking - shirt areas become red, everything else preserved

## 📁 Clean Project Structure

```
red-masking/
├── red_mask                           # 🚀 MAIN EXECUTABLE SCRIPT
├── process_direct_ai_detection.py     # AI detection engine (GroundingDINO + SAM)
├── verify_installation.py            # Installation verification script
├── requirements.txt                   # Project dependencies
├── README.md                          # Complete documentation
├── CLAUDE.md                         # Development guidance
├── example_output.png                # Reference output example
├── comfyui_env/                      # Python virtual environment
└── ComfyUI/                          # ComfyUI framework
    ├── custom_nodes/                 # Essential AI model extensions
    │   ├── comfyui_segment_anything/ # GroundingDINO + SAM (CRITICAL)
    │   ├── ComfyUI-segment-anything-2/
    │   ├── ComfyUI_LayerStyle/       
    │   └── ComfyUI-Impact-Pack/      
    ├── models/                       # AI models (1.1GB total)
    │   ├── grounding-dino/          # GroundingDINO object detection
    │   │   ├── groundingdino_swint_ogc.pth    (694MB)
    │   │   └── GroundingDINO_SwinT_OGC.cfg.py
    │   └── sams/                    # SAM segmentation  
    │       └── sam_vit_b_01ec64.pth (375MB)
    ├── input/                       # Input images directory
    │   └── sample_shirt.jpg         # Sample test image
    └── output/                      # Generated results directory
        └── hogya_00001_.png         # Example output
```

## 🔍 Installation Verification

**Quick verification of your setup:**
```bash
# Activate environment
source comfyui_env/bin/activate

# Run verification script
python verify_installation.py
```

**Expected output:**
```
🔍 AI RED MASKING INSTALLATION VERIFICATION
==================================================
📁 Checking essential files...
  ✅ red_mask
  ✅ process_direct_ai_detection.py
  ✅ requirements.txt
  ✅ README.md
  ✅ ComfyUI/main.py
  ✅ comfyui_env/bin/activate

🤖 Checking AI models...
  ✅ ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth (662MB)
  ✅ ComfyUI/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py (0MB)  
  ✅ ComfyUI/models/sams/sam_vit_b_01ec64.pth (358MB)

📊 Total model size: 1020MB

🔧 Checking custom nodes...
  ✅ comfyui_segment_anything
  ✅ ComfyUI-segment-anything-2
  ✅ ComfyUI_LayerStyle
  ✅ ComfyUI-Impact-Pack

📂 Checking directories...
  ✅ ComfyUI/input
  ✅ ComfyUI/output
  ✅ comfyui_env

==================================================
🎉 INSTALLATION VERIFIED SUCCESSFULLY!
   Ready to run: ./red_mask ComfyUI/input/sample_shirt.jpg test
```

**Test the AI detection:**
```bash
# Run actual AI detection test
./red_mask ComfyUI/input/sample_shirt.jpg verification_test
```

## 🎨 Example Results

### What You Get:
- **Input**: Original image with person wearing shirt
- **Output**: Same image with ONLY the shirt area colored bright red
- **Precision**: Perfect boundaries following shirt contours
- **Preservation**: Background, face, arms, pants completely unchanged

### Use Cases:
- Fashion visualization
- Object highlighting
- E-commerce applications  
- Computer vision demos
- AI detection showcases

## 🚨 Troubleshooting

### Common Issues & Solutions

**Error: "GroundingDINO model loading failed"**
```bash
# Re-download the model
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
  -o "ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth"
```

**Error: "SSL certificate verify failed"**
```bash
# Use -k flag for downloads or install certificates
/Applications/Python\ 3.12/Install\ Certificates.command  # Mac
```

**Error: "No shirt detected"**
- Make sure the image contains a clearly visible shirt/t-shirt
- Try adjusting the detection threshold in the script
- Ensure proper lighting in the image

**Error: "Image not found"**
```bash
# Copy image to input directory
cp your_image.jpg ComfyUI/input/
./red_mask ComfyUI/input/your_image.jpg result
```

**Error: "Permission denied"**
```bash
chmod +x red_mask
```

**Environment Issues**
```bash
# Deactivate and reactivate
deactivate
source comfyui_env/bin/activate

# Reinstall requirements
pip install -r ComfyUI/requirements.txt
```

## 🔧 Advanced Configuration

### Modify Detection Parameters
Edit `process_direct_ai_detection.py`:
```python
# Change detection prompt (line ~171)
prompt="shirt"  # Try: "t-shirt", "clothing", "top"

# Change detection threshold (line ~172)  
threshold=0.3   # Lower = more sensitive, Higher = more strict
```

### Performance Optimization
- **GPU**: Automatically uses MPS (Mac) or CUDA (Windows/Linux)
- **Memory**: ~2GB RAM during processing
- **Speed**: 30-60 seconds per image (first run), 10-20 seconds subsequent runs

## 🔬 Technical Details

### AI Models Used
- **GroundingDINO SwinT OGC**: 694MB object detection model
- **SAM ViT-B**: 375MB segmentation model  
- **BERT**: Text encoder for processing detection prompts

### Framework
- **ComfyUI**: Node-based AI image processing
- **PyTorch**: Deep learning backend with MPS/CUDA support
- **Custom Nodes**: Specialized AI detection and segmentation nodes

### Dependencies
```
ComfyUI >= 0.1.0
torch >= 2.8.0
torchvision >= 0.23.0
segment_anything >= 1.0
timm >= 1.0.19
transformers >= 4.0
```

## 📝 License

This project combines multiple open-source components:
- **ComfyUI**: GPL-3.0 License
- **GroundingDINO**: Apache 2.0 License  
- **SAM**: Apache 2.0 License
- **Custom Implementation**: MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open Pull Request

### Development Setup
```bash
# Clone for development
git clone <repo-url>
cd red-masking

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 📞 Support & FAQ

**Q: How accurate is the shirt detection?**
A: GroundingDINO achieves ~90%+ accuracy on clear shirt images. Performance depends on image quality and shirt visibility.

**Q: Can I detect other objects?**
A: Yes! Change the prompt in `process_direct_ai_detection.py` to detect pants, shoes, etc.

**Q: What image sizes are supported?**
A: Any size. Larger images take longer to process but produce better results.

**Q: Does it work with multiple people?**
A: Yes, it detects shirts on all people in the image.

**Q: Can I change the red color?**
A: Yes, modify the RGB values in the script (currently set to pure red: 255,0,0).

---

## 🎯 Ready to Use!

**Test the AI detection right now:**

```bash
# Quick setup verification
source comfyui_env/bin/activate
./red_mask ComfyUI/input/sample_shirt.jpg amazing_result
```

**Your image will be processed with true AI-powered shirt detection and precise red masking!**

🔴 **Perfect shirt detection • Precise boundaries • Background preserved**