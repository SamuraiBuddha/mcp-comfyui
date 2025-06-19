# ComfyUI MCP Server

A Model Context Protocol (MCP) server that enables Claude to interact with ComfyUI for AI image generation using Stable Diffusion.

## Overview

This MCP server provides a bridge between Claude and ComfyUI, allowing you to:
- Generate images using text prompts
- Execute saved ComfyUI workflows
- Manage the generation queue
- Retrieve generated images
- List available models and workflows

**Special Focus**: Optimized workflows for Crisis Corps logo and branding generation!

## Features

- **Platform Agnostic**: Works with any ComfyUI installation (local, remote, containerized)
- **Simple API**: Straightforward functions for common tasks
- **Workflow Support**: Load and execute complex ComfyUI workflows
- **Queue Management**: Monitor generation progress
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

## Logo Generation Examples

### Generate Crisis Corps Logo
```python
# Using pre-built workflow
result = await execute_workflow(
    workflow_name="crisis_corps_logo.json"
)

# Custom generation
result = await generate_image(
    prompt="Crisis Corps logo, heroic robot emblem, orange and blue, minimalist design, vector style",
    negative_prompt="realistic, photo, 3d render, complex",
    width=1024,
    height=1024,
    steps=35,
    cfg_scale=10
)
```

### Generate Variations
```python
# App icon
icon = await generate_image(
    prompt="Crisis Corps app icon, robot head, rounded square, flat design, orange accent",
    width=512,
    height=512
)

# Banner logo
banner = await generate_image(
    prompt="Crisis Corps banner, horizontal logo, robot silhouettes, emergency orange",
    width=1920,
    height=480
)
```

## Available Tools

### generate_image
Generate an image from a text prompt using the default workflow.

```python
generate_image(
    prompt="A futuristic robot logo for Crisis Corps",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    steps=20,
    cfg_scale=7.0,
    seed=-1  # Random seed
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

### list_models
List available model checkpoints.

```python
list_models()
# Returns: ["sd_xl_base_1.0.safetensors", "dreamshaper_8.safetensors", ...]
```

### interrupt_generation
Stop the current generation.

```python
interrupt_generation()
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
Claude → MCP Server → ComfyUI API
             ↓              ↓
      Configuration    WebSocket
             ↓              ↓
        Return Data ← Generated Images
```

## Error Handling

The server includes comprehensive error handling:
- Connection errors to ComfyUI
- Invalid workflow specifications
- Generation failures
- Timeout handling

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

- [ ] Add LoRA support for logo-specific models
- [ ] Implement ControlNet workflows for consistent shapes
- [ ] Add image-to-image generation for logo variations
- [ ] Support for SDXL specific features
- [ ] Batch processing optimizations
- [ ] Caching for frequently used workflows
- [ ] Auto-background removal for logos
- [ ] SVG conversion support

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic
- Crisis Corps branding examples included with permission
