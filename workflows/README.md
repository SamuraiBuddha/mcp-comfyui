# Crisis Corps Logo Workflows

This directory contains pre-configured ComfyUI workflows optimized for generating Crisis Corps branding assets.

## Available Workflows

### 1. logo_generator.json
**Purpose**: General-purpose logo generation with customizable parameters
- **Best for**: Initial concept exploration
- **Output**: 1024x1024 logos
- **Batch**: Single image
- **Customizable**: Modify the prompt for different concepts

### 2. crisis_corps_logo.json  
**Purpose**: Specifically tuned for Crisis Corps brand identity
- **Best for**: Official logo variations
- **Output**: 1024x1024 logos
- **Batch**: 4 variations per run
- **Features**: Optimized negative prompts for clean design

### 3. robot_emblem.json
**Purpose**: Military-style badge and emblem designs
- **Best for**: Achievement badges, pins, collectibles
- **Output**: 768x768 emblems
- **Batch**: 6 variations per run
- **Style**: Metallic, hexagonal, tactical aesthetics

### 4. text_logo_variations.json
**Purpose**: Typography-focused wordmarks
- **Best for**: Headers, banners, text-only applications  
- **Output**: 1536x512 horizontal logos
- **Batch**: 3 variations per run
- **Style**: Bold, futuristic, stencil-inspired

## Quick Start

1. **Load a workflow in ComfyUI**:
   - Copy the workflow JSON file to your ComfyUI workflows folder
   - Or use through MCP: `execute_workflow("crisis_corps_logo.json")`

2. **Customize prompts** (optional):
   - Edit the positive prompt in node "2"
   - Edit the negative prompt in node "3"
   - Adjust dimensions in node "4"

3. **Generate variations**:
   - Change seed to -1 for random variations
   - Or set specific seeds for reproducible results

## Recommended Models

- **Primary**: SDXL Base 1.0 (included in workflows)
- **Alternative**: DreamShaper XL (more artistic)
- **Logo-specific**: LogoRedmond-LogoLoraForSDXL (if available)

## Tips for Best Results

1. **Keep prompts focused**: Avoid overly complex descriptions
2. **Use style keywords**: "vector", "minimalist", "flat design"
3. **Specify background**: "white background" for clean isolation
4. **Include technical terms**: "scalable", "professional branding"
5. **Batch generate**: Create multiple options to choose from

## Customization Guide

### Changing Colors
Find the positive prompt and modify:
```json
"orange and blue color scheme" → "your colors here"
```

### Adjusting Quality
- **Steps**: 20-25 (draft), 30-40 (final)
- **CFG Scale**: 7-8 (creative), 9-11 (strict)
- **Sampler**: DPM++ 2M Karras (balanced), DPM++ 3M SDE (quality)

### Different Sizes
Modify node "4" (EmptyLatentImage):
```json
"width": 1024,  // Standard square
"width": 1920,  // Banner width
"width": 512,   // Icon size
```

## Integration with MCP

```python
# Execute a pre-made workflow
result = await execute_workflow(
    workflow_name="crisis_corps_logo.json",
    inputs={
        "company_name": "Crisis Corps",
        "style": "heroic"
    }
)

# Or generate custom
result = await generate_image(
    prompt="Crisis Corps logo, your custom design here",
    width=1024,
    height=1024
)
```

## Troubleshooting

**Issue**: Logos look photorealistic
- Add to negative prompt: "photograph, 3d render, realistic"

**Issue**: Text is unreadable
- Use separate text generation workflow
- Or add text in post-processing

**Issue**: Colors are wrong
- Be specific: "#FF6B35 orange" instead of just "orange"
- Check your model's color understanding

**Issue**: Too complex/busy
- Add "minimalist", "simple", "clean" to prompt
- Increase negative prompt strength
