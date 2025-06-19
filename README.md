# ComfyUI MCP Server - Enhanced Edition

A Model Context Protocol (MCP) server that enables Claude to interact with ComfyUI for AI image generation using Stable Diffusion - now with full API control!

## Overview

This enhanced MCP server provides a comprehensive bridge between Claude and ComfyUI, allowing you to:
- Generate images with full control over models, samplers, and schedulers
- Build custom workflows programmatically
- Execute and manage saved ComfyUI workflows
- Upload images for img2img workflows
- List and use LoRAs, embeddings, and custom nodes
- Manage the generation queue
- Retrieve generated images

**Special Focus**: Optimized workflows for Crisis Corps logo and branding generation!

## New Features in v0.2.0

- **Model Swapping**: Change checkpoints on the fly
- **Workflow Builder**: Create workflows programmatically without the UI
- **Advanced Sampling**: Control samplers and schedulers
- **LoRA Support**: List and use LoRA models (coming soon)
- **Node Discovery**: Get all available node types from ComfyUI
- **Image Upload**: Upload images for img2img and ControlNet workflows
- **Queue Management**: Clear queue, check status, interrupt generations
- **Workflow Saving**: Save custom workflows for reuse

## Features

- **Platform Agnostic**: Works with any ComfyUI installation (local, remote, containerized)
- **Full API Access**: Complete control over ComfyUI's capabilities
- **Workflow Support**: Load, execute, build, and save complex workflows
- **Queue Management**: Monitor and control generation progress
- **Flexible Output**: Return images as base64 or file paths
- **Logo Optimized**: Includes pre-built workflows for logo generation

## Prerequisites

- ComfyUI installed and running (see [Setup Guide](setup_comfyui.md))
- Python 3.10+
- MCP SDK

## Installation

```bash
# Clone the repository
git clone https://github.com/SamuraiBuddha/mcp-comfyui.git
cd mcp-comfyui

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Start ComfyUI
```bash
# If installed locally
cd /path/to/ComfyUI
python main.py --listen

# Or use Docker
docker-compose up -d
```

### 2. Configure MCP
```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
# COMFYUI_HOST=localhost
# COMFYUI_PORT=8188
```

### 3. Add to Claude Desktop
```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": ["-m", "mcp_comfyui"],
      "cwd": "/path/to/mcp-comfyui",
      "env": {
        "COMFYUI_HOST": "localhost",
        "COMFYUI_PORT": "8188"
      }
    }
  }
}
```

## Available Tools

### generate_image
Generate an image with full control over all parameters.

```python
generate_image(
    prompt="A futuristic robot logo for Crisis Corps",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    steps=20,
    cfg_scale=7.0,
    seed=-1,  # Random seed
    model="sd_xl_base_1.0.safetensors",
    sampler="euler",
    scheduler="normal"
)
```

### build_workflow
Create a custom workflow programmatically.

```python
build_workflow(
    nodes=[
        {
            "id": "1",
            "type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        {
            "id": "2",
            "type": "CLIPTextEncode",
            "inputs": {"text": "robot logo", "clip": ["1", 1]}
        },
        {
            "id": "3",
            "type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "seed": 42,
                "steps": 20
            }
        }
    ]
)
```

### save_workflow
Save a workflow for future use.

```python
save_workflow(
    name="my_logo_workflow",
    workflow=built_workflow,
    description="Custom workflow for Crisis Corps logos"
)
```

### execute_workflow
Run a saved ComfyUI workflow with custom inputs.

```python
execute_workflow(
    workflow_name="logo_generator",
    inputs={
        "prompt": "Crisis Corps emblem",
        "style": "military insignia"
    }
)
```

### list_models
Get all available model checkpoints.

```python
list_models()
# Returns: ["sd_xl_base_1.0.safetensors", "dreamshaper_8.safetensors", ...]
```

### list_samplers
Get available sampling methods.

```python
list_samplers()
# Returns: ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", ...]
```

### list_schedulers
Get available noise schedulers.

```python
list_schedulers()
# Returns: ["normal", "karras", "exponential", "sgm_uniform", ...]
```

### get_node_types
Discover all available ComfyUI nodes.

```python
get_node_types()
# Returns complete node definitions with inputs/outputs
```

### upload_image
Upload an image for img2img workflows.

```python
upload_image(
    image_path="/path/to/image.png",
    name="reference_image"
)
```

### list_workflows
Get all available workflow files.

```python
list_workflows()
# Returns: ["logo_generator.json", "crisis_corps_logo.json", ...]
```

### get_queue_status
Check the current generation queue.

```python
get_queue_status()
# Returns: {"queue_remaining": 2, "currently_processing": "prompt_123"}
```

### get_history
Retrieve recent generation history.

```python
get_history(limit=10)
# Returns list of recent generations with IDs and parameters
```

### get_image
Retrieve a generated image by ID.

```python
get_image(prompt_id="abc123")
# Returns: base64 encoded image or filepath
```

### interrupt_generation
Stop the current generation.

```python
interrupt_generation()
```

### clear_queue
Clear all pending generations.

```python
clear_queue()
```

## Logo Generation Examples

### Generate Crisis Corps Logo with Different Models
```python
# SDXL for high quality
result = await generate_image(
    prompt="Crisis Corps logo, heroic robot emblem, orange and blue",
    model="sd_xl_base_1.0.safetensors",
    width=1024,
    height=1024,
    steps=35
)

# DreamShaper for stylized look
result = await generate_image(
    prompt="Crisis Corps logo, heroic robot emblem, orange and blue",
    model="dreamshaper_8.safetensors",
    sampler="dpm_2_ancestral",
    scheduler="karras"
)
```

### Build Custom Logo Workflow
```python
# Create a workflow with LoRA for consistent style
workflow = await build_workflow(
    nodes=[
        {"id": "1", "type": "CheckpointLoaderSimple", 
         "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        {"id": "2", "type": "LoraLoader",
         "inputs": {"model": ["1", 0], "clip": ["1", 1], 
                   "lora_name": "logo_style.safetensors",
                   "strength_model": 0.8, "strength_clip": 0.8}},
        {"id": "3", "type": "CLIPTextEncode",
         "inputs": {"text": "Crisis Corps emblem", "clip": ["2", 1]}},
        # ... rest of workflow
    ]
)

# Save for reuse
await save_workflow(
    name="crisis_corps_lora_workflow",
    workflow=workflow,
    description="Logo generation with consistent style LoRA"
)
```

## Pre-Built Workflows

The `workflows/` directory contains optimized workflows for Crisis Corps branding:

1. **logo_generator.json** - General purpose logo creation
2. **crisis_corps_logo.json** - Specific Crisis Corps branding (4 variations)
3. **robot_emblem.json** - Military-style badges and emblems (6 variations)
4. **text_logo_variations.json** - Typography-focused designs

See [workflows/README.md](workflows/README.md) for detailed documentation.

## Brand Guidelines

For consistent Crisis Corps branding, see [examples/brand_guidelines.md](examples/brand_guidelines.md) which includes:
- Color codes (#FF6B35 orange, #004E98 blue)
- Typography guidelines
- Prompt engineering tips
- Style references

## Architecture

```
Claude ↔ MCP Server ↔ ComfyUI API
           ↓             ↓
     Configuration   WebSocket
           ↓             ↓
      Return Data ← Generated Images
```

## Error Handling

The server includes comprehensive error handling:
- Connection errors to ComfyUI
- Invalid workflow specifications
- Generation failures
- Timeout handling
- Model/sampler validation

## Security Notes

- Never expose ComfyUI directly to the internet
- Use API keys if implementing authentication
- Validate all inputs before passing to ComfyUI
- Consider rate limiting for production use

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## TODO

- [ ] Add LoRA support implementation
- [ ] Implement ControlNet workflows for consistent shapes
- [ ] Add image-to-image generation for logo variations
- [ ] Support for SDXL specific features
- [ ] Batch processing optimizations
- [ ] Caching for frequently used workflows
- [ ] Auto-background removal for logos
- [ ] SVG conversion support
- [ ] Custom node support
- [ ] Workflow validation improvements

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic
- Crisis Corps branding examples included with permission
