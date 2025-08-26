# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **minimal AI-powered image processing system** that detects shirt objects in images using **GroundingDINO + SAM** and applies precise red masking ONLY to detected shirt areas. The system is optimized for production deployment with minimal dependencies and instant startup.

## Current Implementation

**✅ MINIMAL PRODUCTION-READY SYSTEM**
- **Main Script**: `minimal_comfyui_detection.py` - Lightweight AI detection engine
- **Executable**: `red_mask_minimal` - Single-command interface
- **Method**: GroundingDINO text-guided object detection + SAM precise segmentation
- **Result**: Perfect red masking only on detected shirt areas
- **Architecture**: Uses ComfyUI nodes directly without server framework overhead

**🔧 Design Philosophy:**
- **Minimal overhead**: Direct AI node usage without web server/GUI components
- **Production ready**: Essential dependencies only (~8 packages vs 50+ in full frameworks)
- **Same AI accuracy**: Identical GroundingDINO + SAM models as full systems
- **Instant startup**: No framework initialization - direct model execution

## Environment Setup

### Prerequisites
- Python 3.12+
- Git
- ~1.1GB storage space (AI models)
- Internet connection for initial downloads

### Installation Steps

```bash
cd "red masking"

# Create minimal virtual environment
python3 -m venv venv_minimal
source venv_minimal/bin/activate  # Mac/Linux
# or venv_minimal\Scripts\activate  # Windows

# Install essential dependencies only
pip install -r requirements_minimal.txt

# Alternative direct installation:
pip install torch torchvision Pillow numpy transformers timm addict yapf opencv-python

# Make executable
chmod +x red_mask_minimal

# Test with sample image
./red_mask_minimal ComfyUI/input/test.png demo_output
```

### Model Requirements

The system automatically loads AI models from ComfyUI structure:
- **GroundingDINO**: `ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth` (694MB)
- **SAM**: `ComfyUI/models/sams/sam_vit_b_01ec64.pth` (375MB) 
- **Config**: `ComfyUI/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py`

If models are missing, the system provides clear download instructions.

## Architecture

### Minimal AI Detection Pipeline
1. **Direct Node Import**: Uses ComfyUI nodes without server framework
2. **GroundingDINO**: Text-guided object detection using prompt "shirt"
3. **SAM**: Precise segmentation masks for detected objects
4. **Red Overlay**: Pure red color applied only to AI-detected areas
5. **No Overhead**: Bypasses web server, APIs, GUI components

### Key Technical Components
- **Node Integration**: Direct usage of `comfyui_segment_anything` nodes
- **Model Management**: Automatic device allocation (MPS/CUDA/CPU)
- **Tensor Processing**: PyTorch-based image and mask manipulation
- **Memory Efficiency**: Minimal overhead compared to full frameworks

## Running the System

### 🚀 Primary Usage (Single Command)

**Simple execution:**
```bash
./red_mask_minimal <image_path> [output_name]
```

**Examples:**
```bash
# Process sample image
./red_mask_minimal ComfyUI/input/test.png

# Process with custom output name
./red_mask_minimal my_photo.jpg custom_result

# Process various formats
./red_mask_minimal shirt.png final_output
```

### 📋 Advanced Usage

**Direct Python execution with parameters:**
```bash
python minimal_comfyui_detection.py image.jpg -o output.png -p "jacket" -t 0.2
```

**Available options:**
- `-o, --output`: Output file path
- `-p, --prompt`: Detection prompt (default: "shirt")
- `-t, --threshold`: Detection threshold (default: 0.3)

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
- **Processing Time**: 10-30 seconds per image (instant startup)
- **Image Support**: All PIL formats (JPG, PNG, BMP, TIFF, WebP, etc.)

## Clean File Structure

```
red-masking/
├── 🚀 red_mask_minimal                # MAIN EXECUTABLE
├── minimal_comfyui_detection.py       # Minimal AI detection engine  
├── requirements_minimal.txt           # Essential dependencies only
├── README.md                          # Complete documentation
├── CLAUDE.md                         # This development guide
├── example_output.png                # Reference result
├── .gitignore                        # Git ignore rules
├── venv_minimal/                     # Virtual environment (setup creates)
└── ComfyUI/                          # ComfyUI nodes and models
    ├── custom_nodes/                 # AI model extensions
    │   └── comfyui_segment_anything/ # GroundingDINO + SAM (CRITICAL)
    ├── models/                       # AI models (1.1GB total)
    │   ├── grounding-dino/          # Object detection
    │   │   ├── groundingdino_swint_ogc.pth    (694MB)
    │   │   └── GroundingDINO_SwinT_OGC.cfg.py
    │   └── sams/                    # Segmentation
    │       └── sam_vit_b_01ec64.pth (375MB)
    ├── input/                       # Input images directory
    │   └── test.png                 # Sample test image
    └── output/                      # Generated results directory
```

## Technical Implementation

### Core Implementation Details

**File**: `minimal_comfyui_detection.py`

**Key Classes and Methods:**
```python
class MinimalComfyUIDetector:
    def load_models()                    # Direct ComfyUI node loading
    def detect_and_segment()             # Combined detection + segmentation
    def apply_red_masking()              # Red overlay application  
    def process_image()                  # Complete pipeline
```

**Direct Node Usage:**
```python
# Import ComfyUI nodes without server framework
from node import SAMModelLoader, GroundingDinoModelLoader, GroundingDinoSAMSegment

# Direct model loading
groundingdino_loader = GroundingDinoModelLoader()
sam_loader = SAMModelLoader()

# Combined detection + segmentation in single call
grounding_sam_segment = GroundingDinoSAMSegment()
result = grounding_sam_segment.main(
    prompt="shirt", threshold=0.3,
    sam_model=sam_model, grounding_dino_model=groundingdino_model,
    image=image_tensor
)
```

### Minimal System Advantages
- **Size**: 1.1GB total vs 2.4GB+ full frameworks
- **Startup**: Instant vs 30+ seconds framework initialization
- **Dependencies**: ~8 essential packages vs 50+ in full systems
- **Memory**: Lower overhead due to direct node usage
- **Accuracy**: Identical AI models - same detection quality
- **Deployment**: Perfect for production, CI/CD, automation

## Troubleshooting

### Model Loading Issues
- **Check models exist**: `ls -la ComfyUI/models/grounding-dino/ ComfyUI/models/sams/`
- **Re-download if missing**: Use curl commands in README
- **Verify file sizes**: GroundingDINO ~694MB, SAM ~375MB

### Node Loading Issues  
- **Check custom node**: `ls -la ComfyUI/custom_nodes/comfyui_segment_anything/`
- **Reinstall if missing**: `git clone https://github.com/storyicon/comfyui_segment_anything.git`
- **Environment activation**: `source venv_minimal/bin/activate`

### Detection Issues
- **Try different prompts**: "t-shirt", "jacket", "blazer", "clothing"
- **Adjust threshold**: Lower values (0.2) = more sensitive, Higher (0.5) = more strict
- **Image quality**: Ensure clear, well-lit garments
- **Multiple objects**: System detects all matching objects in image

### Common Issues
- **SSL certificate errors**: Use `-k` flag in curl downloads
- **Permission errors**: `chmod +x red_mask_minimal`
- **Memory errors**: Reduce image size or ensure sufficient RAM
- **Import errors**: Check virtual environment activation

## Development Guidelines

### Modifying Detection Parameters

**Edit `minimal_comfyui_detection.py`:**
```python
# Line ~98: Change detection prompt
prompt="shirt"  # Try: "t-shirt", "jacket", "blazer", "clothing"

# Line ~98: Change detection threshold
threshold=0.3   # Lower = more sensitive, Higher = more strict

# Line ~113: Change red color values
red_overlay[:, :, :, 0] = 1.0  # Red channel (0.0-1.0)
red_overlay[:, :, :, 1] = 0.0  # Green channel
red_overlay[:, :, :, 2] = 0.0  # Blue channel
```

### Adding New Features
- **Keep minimal**: Avoid heavy dependencies that increase footprint
- **Maintain compatibility**: Ensure ComfyUI node structure compatibility
- **Test thoroughly**: Verify with various image types and sizes
- **Document changes**: Update README and this file for modifications

### Code Standards
- **Focus on essentials**: Only include features needed for AI detection
- **Optimize for deployment**: Consider production use cases
- **Maintain performance**: Keep startup time minimal
- **Error handling**: Provide clear error messages for common issues

## Performance Optimization

### Memory Management
- **Model loading**: Models loaded once per execution
- **Image processing**: Efficient tensor operations
- **GPU utilization**: Automatic device selection for optimal performance
- **Memory cleanup**: Proper tensor disposal after processing

### Processing Efficiency
- **Direct node calls**: Bypass unnecessary framework layers
- **Batch processing**: Single image per execution (scriptable for batches)
- **Resolution handling**: Automatic scaling for optimal processing
- **Format support**: Native PIL format handling for broad compatibility

## Production Deployment

### Recommended Setup
1. **Use minimal version** for all production deployments
2. **Install in isolated environment** with `venv_minimal`
3. **Pre-download models** during deployment, not runtime
4. **Script for batch processing** if needed
5. **Monitor memory usage** for concurrent executions

### CI/CD Integration
```bash
# Example CI/CD pipeline step
source venv_minimal/bin/activate
./red_mask_minimal input_image.jpg output_result
# Upload output_result.png to target destination
```

### Docker Considerations
- **Base image**: Use Python 3.12 slim
- **Model caching**: Include models in container or mount volume
- **Dependencies**: Install requirements_minimal.txt only
- **Size optimization**: Multi-stage build to minimize final image

---

**Current Status**: ✅ Production-ready minimal AI detection system  
**Recommendation**: Use this version for all production deployments  
**Performance**: Verified working with instant startup and same AI accuracy  
**Last Updated**: August 2025  
**Testing**: Verified on macOS with MPS acceleration