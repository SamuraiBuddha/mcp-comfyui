# ComfyUI MCP Server

A Model Context Protocol (MCP) server that enables Claude to interact with ComfyUI for AI image generation using Stable Diffusion.

## Overview

This MCP server provides a bridge between Claude and ComfyUI, allowing you to:
- Generate images using text prompts
- Execute saved ComfyUI workflows
- Manage the generation queue
- Retrieve generated images
- List available models and workflows

## Features

- **Platform Agnostic**: Works with any ComfyUI installation (local, remote, containerized)
- **Simple API**: Straightforward functions for common tasks
- **Workflow Support**: Load and execute complex ComfyUI workflows
- **Queue Management**: Monitor generation progress
- **Flexible Output**: Return images as base64 or file paths

## Prerequisites

- ComfyUI installed and running (see [ComfyUI Installation](https://github.com/comfyanonymous/ComfyUI))
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

## Configuration

### Environment Variables

```bash
# ComfyUI connection settings
COMFYUI_HOST=localhost  # ComfyUI server host
COMFYUI_PORT=8188       # ComfyUI server port

# Output settings
COMFYUI_OUTPUT_DIR=/path/to/comfyui/output  # Where ComfyUI saves images
MCP_RETURN_FORMAT=base64  # 'base64' or 'filepath'

# Optional authentication
COMFYUI_API_KEY=your_api_key_if_needed
```

### Claude Desktop Configuration

Add to your Claude Desktop config:

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
# Returns: ["logo_generator.json", "texture_creator.json", ...]
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

## Usage Examples

### Generate a Logo

```python
# Simple logo generation
result = await generate_image(
    prompt="minimalist robot head logo, clean lines, tech startup style, orange and blue",
    negative_prompt="complex, detailed, photorealistic",
    width=1024,
    height=1024,
    steps=30
)
```

### Use a Custom Workflow

```python
# Execute a pre-saved workflow for consistent results
result = await execute_workflow(
    workflow_name="brand_logo_generator",
    inputs={
        "company_name": "Crisis Corps",
        "style": "modern",
        "colors": ["#FF6B35", "#004E98"]
    }
)
```

### Batch Generation

```python
# Generate multiple variations
for i in range(4):
    await generate_image(
        prompt="Crisis Corps robot mascot, friendly, heroic",
        seed=i * 1000  # Different seeds for variations
    )
```

## Workflow Management

Workflows should be saved in ComfyUI's workflow directory. The MCP server can load and execute any valid ComfyUI workflow JSON file.

### Creating Logo-Specific Workflows

1. Design your workflow in ComfyUI's web interface
2. Save it with a descriptive name (e.g., `logo_generator.json`)
3. The workflow will be available via `list_workflows()`

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

- [ ] Add LoRA support
- [ ] Implement ControlNet workflows
- [ ] Add image-to-image generation
- [ ] Support for SDXL specific features
- [ ] Batch processing optimizations
- [ ] Caching for frequently used workflows

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic
