"""Utility functions for MCP ComfyUI server."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image
import io


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def validate_workflow(workflow: Dict[str, Any]) -> bool:
    """Validate a ComfyUI workflow structure."""
    if not isinstance(workflow, dict):
        return False
    
    # Basic validation - each node should have class_type and inputs
    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            return False
        if "class_type" not in node_data:
            return False
        if "inputs" not in node_data:
            return False
    
    return True


def merge_workflow_inputs(workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user inputs into a workflow template."""
    # This is a simplified version - actual implementation would need
    # to understand the workflow structure and input mappings
    modified_workflow = workflow.copy()
    
    # Example: Look for text inputs and replace them
    for node_id, node_data in modified_workflow.items():
        if node_data.get("class_type") == "CLIPTextEncode":
            if "prompt" in inputs and "positive" in str(node_data):
                node_data["inputs"]["text"] = inputs["prompt"]
            elif "negative_prompt" in inputs and "negative" in str(node_data):
                node_data["inputs"]["text"] = inputs["negative_prompt"]
    
    return modified_workflow


def extract_image_outputs(history_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Extract image outputs from ComfyUI history data."""
    outputs = []
    
    # Navigate through the history structure to find saved images
    if "outputs" in history_data:
        for node_id, node_output in history_data["outputs"].items():
            if "images" in node_output:
                for image_info in node_output["images"]:
                    outputs.append({
                        "node_id": node_id,
                        "filename": image_info.get("filename"),
                        "subfolder": image_info.get("subfolder", ""),
                        "type": image_info.get("type", "output"),
                    })
    
    return outputs
