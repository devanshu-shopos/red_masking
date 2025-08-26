# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered image processing workflow that detects shirt objects in images using **GroundingDINO + SAM** and applies precise red masking ONLY to detected shirt areas. The project offers **two versions**: a **minimal standalone** version (recommended) and a **full ComfyUI** version for advanced use cases.

## Two Implementation Approaches

### 🎯 **Minimal Version (Primary - Recommended)**
- **File**: `minimal_comfyui_detection.py`
- **Executable**: `red_mask_minimal`
- **Size**: ~1.1GB (60% smaller than full)
- **Dependencies**: Essential AI models only
- **Startup**: Instant (no server overhead)
- **Use case**: Production, deployment, quick testing

### 🔧 **Full ComfyUI Version (Secondary)**
- **File**: `process_direct_ai_detection.py` 
- **Executable**: `red_mask`
- **Size**: ~2.4GB (complete framework)
- **Dependencies**: Full ComfyUI ecosystem
- **Startup**: 30+ seconds (framework initialization)
- **Use case**: Research, advanced workflows, ComfyUI ecosystem

## Current Working Implementation

**✅ BOTH VERSIONS WORKING**
- **Minimal**: Uses ComfyUI nodes directly without server framework
- **Full**: Uses complete ComfyUI system with all components
- **Method**: GroundingDINO text-guided object detection + SAM precise segmentation
- **Result**: Perfect red masking only on detected shirt areas

**❌ REMOVED: All faulty approaches including:**
- Color-based detection methods
- Geometric overlay approaches
- Brightness-based targeting
- Multi-layer enhancement techniques

## Environment Setup Options

### Option 1: Minimal Setup (Recommended)

**Prerequisites:**
- Python 3.12+
- Git
- ~1.1GB storage space

**Quick Installation:**
```bash
cd "red masking"
python3 -m venv venv_minimal
source venv_minimal/bin/activate

# Core dependencies only
pip install torch torchvision Pillow numpy transformers timm addict yapf opencv-python

# Download AI models (1GB)
mkdir -p models
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
  -o "models/groundingdino_swint_ogc.pth"
curl -k -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  -o "models/sam_vit_b_01ec64.pth"

chmod +x red_mask_minimal
./red_mask_minimal test.png
```

### Option 2: Full ComfyUI Setup

**Prerequisites:**
- Python 3.12+
- Git  
- ~2.4GB storage space

**Complete Installation:**
```bash
cd "red masking"
python3 -m venv comfyui_env
source comfyui_env/bin/activate

# Install ComfyUI framework
git clone https://github.com/comfyanonymous/ComfyUI.git
pip install -r ComfyUI/requirements.txt

# Install CRITICAL custom nodes
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

chmod +x red_mask  
./red_mask ComfyUI/input/test.png
```

## Architecture Comparison

### Minimal Version Architecture
**Direct AI Node Usage:**
1. **Direct Import**: Imports ComfyUI nodes directly without server
2. **GroundingDINO**: Text-guided object detection using prompt "shirt"
3. **SAM**: Precise segmentation masks for detected objects
4. **Red Overlay**: Pure red color applied only to AI-detected areas
5. **No Overhead**: Bypasses web server, APIs, GUI components

**Key Components:**
- **Node Integration**: Uses existing `comfyui_segment_anything` nodes
- **Direct Execution**: No ComfyUI server initialization  
- **Model Management**: Automatic device allocation (MPS/CUDA/CPU)
- **Tensor Processing**: PyTorch-based image and mask manipulation

### Full Version Architecture
**Complete ComfyUI Pipeline:**
1. **Framework Init**: Full ComfyUI server and node system
2. **Custom Nodes**: All 4 custom node packages loaded
3. **Model Loading**: Through ComfyUI model management system
4. **Detection Pipeline**: Same GroundingDINO + SAM detection
5. **Server Overhead**: Full web server and API system loaded

## Running the Workflows

### 🚀 Minimal Version (Primary Method)

**Simple execution:**
```bash
./red_mask_minimal <image_path> [output_name]
```

**Examples:**
```bash
# Quick test
./red_mask_minimal test.png

# Custom output name
./red_mask_minimal my_photo.jpg custom_result

# Direct Python with options
python minimal_comfyui_detection.py image.jpg -o output.png -p "jacket" -t 0.2
```

### 🔧 Full ComfyUI Version

**Complete framework execution:**
```bash
./red_mask <image_path> [output_name]

# Example
./red_mask ComfyUI/input/test.png full_result
```

### ✅ Installation Verification

**Minimal version:**
```bash
source venv_minimal/bin/activate
./red_mask_minimal test.png verification
```

**Full version:**
```bash
source comfyui_env/bin/activate  
python verify_installation.py
./red_mask ComfyUI/input/test.png verification
```

## Configuration Options

### AI Detection Parameters (Both Versions)
- **Detection Prompt**: "shirt" (customizable)
- **Detection Threshold**: 0.3 (30% confidence minimum)
- **Model Loading**: GroundingDINO SwinT OGC + SAM ViT-B
- **Red Overlay**: Pure red (RGB: 255, 0, 0)
- **Processing**: Binary masking - shirt = red, everything else = original

### Performance Settings
- **GPU Support**: Auto-detects MPS (Mac), CUDA (PC), CPU fallback
- **Memory Usage**: ~2GB during processing
- **Processing Time**: 
  - Minimal: 10-20 seconds (no server overhead)
  - Full: 30-60 seconds (framework initialization)
- **Image Support**: All PIL formats (JPG, PNG, etc.)

## File Structure

```
red-masking/
├── 🚀 red_mask_minimal                # MINIMAL EXECUTABLE (Recommended)
├── minimal_comfyui_detection.py       # Minimal AI engine
├── red_mask                           # Full ComfyUI executable  
├── process_direct_ai_detection.py     # Full AI engine
├── requirements_minimal.txt           # Minimal dependencies
├── requirements.txt                   # Full dependencies
├── verify_installation.py            # Verification utility
├── README.md                          # Complete documentation
├── CLAUDE.md                         # This file
├── example_output.png                # Reference result
├── models/                           # Minimal version models
│   ├── groundingdino_swint_ogc.pth   # Object detection (694MB)
│   ├── GroundingDINO_SwinT_OGC.cfg.py
│   └── sam_vit_b_01ec64.pth         # Segmentation (375MB)
├── venv_minimal/                     # Minimal environment
├── comfyui_env/                      # Full environment (optional)
└── ComfyUI/                          # Full framework (optional)
    ├── custom_nodes/                 # AI model extensions
    │   ├── comfyui_segment_anything/ # GroundingDINO + SAM (CRITICAL)
    │   ├── ComfyUI-segment-anything-2/
    │   ├── ComfyUI_LayerStyle/
    │   └── ComfyUI-Impact-Pack/
    ├── models/                       # ComfyUI model directories
    ├── input/                        # Input images  
    └── output/                       # Generated results
```

## Technical Implementation Notes

### Minimal Version Implementation
**File**: `minimal_comfyui_detection.py`

**Key Classes and Methods:**
```python
class MinimalComfyUIDetector:
    def load_models()                    # Direct ComfyUI node loading
    def detect_and_segment()             # Combined detection + segmentation  
    def apply_red_masking()              # Red overlay application
    def process_image()                  # Complete pipeline
```

**Node Usage:**
```python
# Direct ComfyUI node imports (no server)
from node import SAMModelLoader, GroundingDinoModelLoader, GroundingDinoSAMSegment

# Model loading
groundingdino_loader = GroundingDinoModelLoader()
sam_loader = SAMModelLoader()

# Detection + segmentation in one call
grounding_sam_segment = GroundingDinoSAMSegment()
result = grounding_sam_segment.main(
    prompt="shirt", threshold=0.3,
    sam_model=sam_model, grounding_dino_model=groundingdino_model,
    image=image_tensor
)
```

### Full Version Implementation  
**File**: `process_direct_ai_detection.py`

Uses complete ComfyUI framework with all components loaded.

## Performance Comparison

| Aspect | Minimal Version | Full Version |
|--------|----------------|--------------|
| **Size** | 1.1GB | 2.4GB |
| **Startup** | Instant | 30+ seconds |
| **Memory** | Lower overhead | Higher overhead |
| **Dependencies** | Essential only | Complete ecosystem |
| **Accuracy** | Identical AI models | Identical AI models |
| **Use Case** | Production/deployment | Research/advanced |

## Troubleshooting

### Minimal Version Issues
- **Node loading fails**: Ensure ComfyUI directory exists or install standalone dependencies
- **Model not found**: Check models/ directory has required .pth files
- **Import errors**: Activate venv_minimal and install requirements_minimal.txt

### Full Version Issues  
- **ComfyUI initialization fails**: Run verify_installation.py  
- **Custom nodes missing**: Re-install comfyui_segment_anything node
- **Model paths wrong**: Check ComfyUI/models/ directory structure

### Common Issues
- **SSL certificate errors**: Use `-k` flag in curl downloads
- **No objects detected**: Try lower threshold (0.2) or different prompt
- **Memory errors**: Reduce image size or ensure sufficient RAM

## Development Guidelines

### Modifying Detection Parameters

**Minimal version** - Edit `minimal_comfyui_detection.py`:
```python
# Line ~98: Change detection prompt
prompt="shirt"  # Try: "t-shirt", "jacket", "blazer", "clothing"

# Line ~98: Change detection threshold
threshold=0.3   # Lower = more sensitive, Higher = more strict
```

**Full version** - Edit `process_direct_ai_detection.py`:
```python  
# Line ~171-172: Same parameter modification
```

### Adding New Features
- **Minimal**: Keep lightweight, avoid heavy dependencies
- **Full**: Can leverage complete ComfyUI ecosystem
- **Both**: Maintain same AI detection accuracy
- **Testing**: Test both versions before committing

### Code Standards
- Keep minimal version focused on essential functionality
- Use full version for experimental/advanced features
- Maintain backward compatibility for both versions
- Document performance differences

## Version Selection Guidelines

**Use Minimal Version When:**
- Deploying to production
- Need fast startup times
- Want minimal dependencies  
- Simple deployment/installation
- Resource-constrained environments

**Use Full Version When:**
- Need ComfyUI GUI/web interface
- Developing advanced workflows
- Require additional ComfyUI nodes
- Research/experimentation
- Integration with broader ComfyUI ecosystem

---

**Current Status**: ✅ Both versions production ready with true AI detection  
**Recommendation**: Use minimal version unless you specifically need ComfyUI framework features  
**Last Updated**: August 2025  
**Performance**: Verified working on macOS with MPS acceleration