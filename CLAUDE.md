# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered image processing workflow that detects shirt objects in images using **GroundingDINO + SAM** and applies precise red masking ONLY to detected shirt areas. The system uses true AI-based object detection and segmentation - no geometric overlays or approximations.

## Current Working Implementation

**✅ WORKING APPROACH: True AI Detection**
- **Main Script**: `process_direct_ai_detection.py` - Direct AI-powered shirt detection
- **Executable**: `red_mask` - Single-command interface 
- **Method**: GroundingDINO text-guided object detection + SAM precise segmentation
- **Result**: Perfect red masking only on detected shirt areas

**❌ REMOVED: All faulty approaches including:**
- Color-based detection methods
- Geometric overlay approaches  
- Brightness-based targeting
- Multi-layer enhancement techniques
- Any method that doesn't use true AI detection

## Environment Setup

### Prerequisites
- Python 3.12+
- Git
- ~1.5GB storage space for AI models
- Internet connection for model downloads

### Complete Installation Steps

1. **Create and activate virtual environment:**
```bash
cd "/path/to/red masking"
python3 -m venv comfyui_env
source comfyui_env/bin/activate  # Mac/Linux
# or comfyui_env\Scripts\activate  # Windows
```

2. **Install ComfyUI framework:**
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
pip install -r ComfyUI/requirements.txt
```

3. **Install CRITICAL custom nodes (order matters):**
```bash
cd ComfyUI/custom_nodes

# MOST IMPORTANT: GroundingDINO + SAM integration
git clone https://github.com/storyicon/comfyui_segment_anything.git

# Supporting nodes
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git

cd ../..
```

4. **Install Python dependencies:**
```bash
pip install segment_anything timm addict yapf platformdirs
pip install -r ComfyUI/custom_nodes/ComfyUI_LayerStyle/requirements.txt
pip install -r requirements.txt
```

5. **Download AI models (CRITICAL - 1GB+ download):**
```bash
mkdir -p ComfyUI/models/grounding-dino ComfyUI/models/sams

# GroundingDINO model (694MB) - Object Detection
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
  -o "ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth"

# GroundingDINO config
curl -k -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py" \
  -o "ComfyUI/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py"

# SAM model (375MB) - Segmentation  
curl -k -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  -o "ComfyUI/models/sams/sam_vit_b_01ec64.pth"
```

6. **Make executable and test:**
```bash
chmod +x red_mask
./red_mask ComfyUI/input/sample_shirt.jpg test
```

## Architecture

**AI Detection Pipeline:**
1. **GroundingDINO**: Text-guided object detection using prompt "shirt"
2. **SAM (Segment Anything)**: Precise segmentation masks for detected objects  
3. **Red Overlay**: Pure red color applied only to AI-detected areas
4. **Preservation**: All non-shirt areas remain completely unchanged

**Key Technical Components:**
- **Node Integration**: `comfyui_segment_anything` provides GroundingDINO + SAM nodes
- **Direct Import**: Bypasses full ComfyUI server initialization for faster processing
- **Model Management**: Automatic model loading with proper device allocation (MPS/CUDA/CPU)
- **Tensor Processing**: PyTorch-based image and mask manipulation

## Running the Workflow

### 🚀 Single-Command Usage (Primary Method)

**Simple execution:**
```bash
./red_mask <image_path> [output_name]
```

**Examples:**
```bash
# Test with sample image
./red_mask ComfyUI/input/sample_shirt.jpg test_output

# Process your own image  
./red_mask /path/to/photo.jpg my_result

# Quick test with custom name
./red_mask shirt.png hogya
```

### 📋 Manual Execution

**Direct Python execution:**
```bash
source comfyui_env/bin/activate
python process_direct_ai_detection.py <image_path> [output_name]
```

### ✅ Installation Verification

**Check setup:**
```bash
source comfyui_env/bin/activate
python verify_installation.py
```

## Configuration

### AI Detection Parameters
- **Detection Prompt**: "shirt" (customizable in script)
- **Detection Threshold**: 0.3 (30% confidence minimum)
- **Model Loading**: GroundingDINO SwinT OGC + SAM ViT-B
- **Red Overlay**: Pure red (RGB: 255, 0, 0)
- **Processing**: Binary masking - shirt = red, everything else = original

### Performance Settings
- **GPU Support**: Auto-detects MPS (Mac), CUDA (PC), CPU fallback
- **Memory Usage**: ~2GB during processing  
- **Processing Time**: 30-60 seconds first run, 10-20 seconds subsequent runs
- **Image Support**: All PIL formats (JPG, PNG, etc.)

## Clean Directory Structure

```
red-masking/
├── red_mask                           # 🚀 MAIN EXECUTABLE
├── process_direct_ai_detection.py     # AI detection engine (GroundingDINO + SAM)
├── verify_installation.py            # Installation verification
├── requirements.txt                   # Project dependencies
├── README.md                          # Complete documentation
├── CLAUDE.md                         # This file
├── example_output.png                # Reference output
├── comfyui_env/                      # Virtual environment
└── ComfyUI/                          # ComfyUI framework
    ├── custom_nodes/                 # Essential AI extensions
    │   ├── comfyui_segment_anything/ # GroundingDINO + SAM (CRITICAL)
    │   ├── ComfyUI-segment-anything-2/
    │   ├── ComfyUI_LayerStyle/       
    │   └── ComfyUI-Impact-Pack/      
    ├── models/                       # AI models (1.1GB total)
    │   ├── grounding-dino/          # Object detection
    │   │   ├── groundingdino_swint_ogc.pth    (694MB)
    │   │   └── GroundingDINO_SwinT_OGC.cfg.py
    │   └── sams/                    # Segmentation
    │       └── sam_vit_b_01ec64.pth (375MB)
    ├── input/                       # Input images
    │   └── sample_shirt.jpg         # Test image
    └── output/                      # Results
        └── hogya_00001_.png         # Example output
```

## Technical Implementation Notes

### Key Files and Functions

**`process_direct_ai_detection.py`** - Core AI detection engine:
- Direct node imports for faster execution
- GroundingDINO + SAM integration
- Tensor-based image processing
- Error handling and model loading

**`red_mask`** - Shell wrapper script:
- Virtual environment activation
- Argument passing to Python script  
- User-friendly interface

### Node Classes Used
```python
# From comfyui_segment_anything custom node:
SAMModelLoader              # Loads SAM segmentation model
GroundingDinoModelLoader    # Loads GroundingDINO detection model  
GroundingDinoSAMSegment     # Combined detection + segmentation
```

### Detection Workflow
```python
# 1. Load models
groundingdino_loader = GroundingDinoModelLoader()
sam_loader = SAMModelLoader()

# 2. Run detection
grounding_sam_segment = GroundingDinoSAMSegment()
detection_result = grounding_sam_segment.main(
    prompt="shirt",           # Text-guided detection
    threshold=0.3,           # Confidence threshold
    sam_model=sam_model,     # Segmentation model
    grounding_dino_model=groundingdino_model,  # Detection model
    image=image_tensor       # Input image
)

# 3. Apply red masking to detected areas only
```

## Troubleshooting

### Common Issues
- **SSL certificate errors**: Use `-k` flag in curl downloads
- **Model loading fails**: Re-download models with correct paths
- **Node not found**: Ensure `comfyui_segment_anything` is properly installed
- **Memory errors**: Reduce image size or ensure sufficient RAM

### Debugging
- Check `python verify_installation.py` output
- Verify model files exist and have correct sizes
- Ensure virtual environment is activated
- Check ComfyUI custom node loading

## Development Guidelines

### Code Modification
- **Detection parameters**: Edit `process_direct_ai_detection.py` line ~171-172
- **Color changes**: Modify RGB values in red overlay section
- **Model updates**: Update model URLs and paths as needed

### Adding Features  
- Keep the core AI detection pipeline intact
- Add new functionality as post-processing steps
- Maintain compatibility with existing node structure
- Test thoroughly before committing changes

---

**Current Status**: ✅ Production ready with true AI detection
**Last Updated**: August 2025
**Performance**: Verified working on macOS with MPS acceleration