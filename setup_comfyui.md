# Setting Up ComfyUI for Logo Generation

## Quick Install Guide

### Option 1: Standalone Installation

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Download models (at minimum, you need one)
# Option A: Download SDXL Base (recommended for logos)
wget -P models/checkpoints/ https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Run ComfyUI
python main.py --listen
```

### Option 2: Docker Installation

```yaml
# docker-compose.yml
version: '3.8'

services:
  comfyui:
    image: yanwk/comfyui-boot:latest
    container_name: comfyui
    ports:
      - "8188:8188"
    volumes:
      - ./models:/home/runner/ComfyUI/models
      - ./output:/home/runner/ComfyUI/output
      - ./workflows:/home/runner/ComfyUI/workflows
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### Option 3: RunPod/Cloud GPU

1. Create RunPod account
2. Deploy "ComfyUI" template
3. Access via provided URL
4. Upload workflows via web interface

## Recommended Models for Logo Generation

### Essential Models

1. **SDXL Base 1.0** (6.9GB)
   - Best general-purpose model
   - Great for clean, professional designs
   ```bash
   wget -P models/checkpoints/ https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
   ```

2. **DreamShaper XL** (6.9GB) - Optional
   - More artistic/stylized results
   - Good for game assets

### Useful LoRAs for Logos

1. **Logo LoRA**
   - Specifically trained on logo designs
   - Place in `models/loras/`

2. **Vector Art LoRA**
   - Clean vector-style outputs
   - Helps with scalability

## ComfyUI Configuration

### settings.json
```json
{
  "api": {
    "enable_api": true,
    "enable_cors": true
  },
  "ui": {
    "default_workflow": "workflows/crisis_corps_logo.json"
  }
}
```

### Performance Settings

For logo generation (smaller batches, higher quality):
```bash
python main.py --listen --highvram --gpu-only
```

## Testing Your Setup

1. **Access ComfyUI**: http://localhost:8188
2. **Load test workflow**: Use `logo_generator.json`
3. **Generate test image**: Queue Prompt
4. **Verify API**: http://localhost:8188/system_stats

## MCP Integration

Once ComfyUI is running:

```bash
# Set environment variables
export COMFYUI_HOST=localhost
export COMFYUI_PORT=8188

# Test MCP connection
cd mcp-comfyui
python -m mcp_comfyui
```

## Optimization Tips

### For Quality
- Use SDXL models (better text understanding)
- Higher step counts (30-40)
- Multiple batches for variety

### For Speed  
- Use SD 1.5 models if quality acceptable
- Lower step counts (20-25)
- Smaller resolutions (512x512)

### For Consistency
- Save good seeds
- Use ControlNet for shape consistency
- Create workflow templates

## Troubleshooting

**GPU not detected:**
```bash
nvidia-smi  # Check CUDA availability
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory:**
- Use `--lowvram` or `--cpu` flags
- Reduce batch size
- Lower resolution

**API not responding:**
- Check firewall settings
- Ensure `--listen` flag is used
- Verify port 8188 is free

**Slow generation:**
- Check model is on GPU
- Reduce image dimensions
- Use faster samplers (Euler a)
