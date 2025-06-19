"""MCP server implementation for ComfyUI integration."""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .comfyui_client import ComfyUIClient
from .utils import encode_image_to_base64, validate_workflow

logger = logging.getLogger(__name__)


class ComfyUIServer:
    """MCP Server for ComfyUI integration."""

    def __init__(self):
        self.server = Server("mcp-comfyui")
        self.client: Optional[ComfyUIClient] = None
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="generate_image",
                    description="Generate an image from a text prompt",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Positive prompt"},
                            "negative_prompt": {"type": "string", "description": "Negative prompt"},
                            "width": {"type": "integer", "default": 512},
                            "height": {"type": "integer", "default": 512},
                            "steps": {"type": "integer", "default": 20},
                            "cfg_scale": {"type": "number", "default": 7.0},
                            "seed": {"type": "integer", "default": -1},
                        },
                        "required": ["prompt"],
                    },
                ),
                types.Tool(
                    name="execute_workflow",
                    description="Execute a saved ComfyUI workflow",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "workflow_name": {"type": "string", "description": "Name of the workflow file"},
                            "inputs": {"type": "object", "description": "Workflow input parameters"},
                        },
                        "required": ["workflow_name"],
                    },
                ),
                types.Tool(
                    name="list_workflows",
                    description="List available workflow files",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_queue_status",
                    description="Get the current queue status",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_history",
                    description="Get generation history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 10},
                        },
                    },
                ),
                types.Tool(
                    name="get_image",
                    description="Retrieve a generated image",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt_id": {"type": "string", "description": "ID of the generation"},
                        },
                        "required": ["prompt_id"],
                    },
                ),
                types.Tool(
                    name="list_models",
                    description="List available model checkpoints",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="interrupt_generation",
                    description="Interrupt the current generation",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution."""
            if not self.client:
                raise RuntimeError("ComfyUI client not initialized")

            if name == "generate_image":
                result = await self._generate_image(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "execute_workflow":
                result = await self._execute_workflow(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "list_workflows":
                workflows = await self._list_workflows()
                return [types.TextContent(type="text", text=json.dumps(workflows, indent=2))]

            elif name == "get_queue_status":
                status = await self.client.get_queue()
                return [types.TextContent(type="text", text=json.dumps(status, indent=2))]

            elif name == "get_history":
                limit = arguments.get("limit", 10) if arguments else 10
                history = await self.client.get_history(limit)
                return [types.TextContent(type="text", text=json.dumps(history, indent=2))]

            elif name == "get_image":
                if not arguments or "prompt_id" not in arguments:
                    raise ValueError("prompt_id is required")
                image_data = await self._get_image(arguments["prompt_id"])
                return [types.TextContent(type="text", text=json.dumps(image_data, indent=2))]

            elif name == "list_models":
                models = await self.client.get_models()
                return [types.TextContent(type="text", text=json.dumps(models, indent=2))]

            elif name == "interrupt_generation":
                result = await self.client.interrupt()
                return [types.TextContent(type="text", text=json.dumps({"interrupted": result}))]

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _generate_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an image using the default workflow."""
        # Build a simple txt2img workflow
        workflow = self._build_default_workflow(
            prompt=args.get("prompt", ""),
            negative_prompt=args.get("negative_prompt", ""),
            width=args.get("width", 512),
            height=args.get("height", 512),
            steps=args.get("steps", 20),
            cfg_scale=args.get("cfg_scale", 7.0),
            seed=args.get("seed", -1),
        )

        prompt_id = await self.client.queue_prompt(workflow)
        
        # Wait for completion
        result = await self.client.wait_for_completion(prompt_id)
        
        return {
            "prompt_id": prompt_id,
            "status": "completed",
            "outputs": result,
        }

    async def _execute_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a saved workflow."""
        workflow_name = args.get("workflow_name")
        if not workflow_name:
            raise ValueError("workflow_name is required")

        # Load workflow from file
        workflow_path = Path(os.getenv("COMFYUI_WORKFLOWS_DIR", "./workflows")) / workflow_name
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_name}")

        with open(workflow_path, "r") as f:
            workflow = json.load(f)

        # Apply any input overrides
        inputs = args.get("inputs", {})
        # TODO: Implement workflow input mapping

        prompt_id = await self.client.queue_prompt(workflow)
        result = await self.client.wait_for_completion(prompt_id)

        return {
            "prompt_id": prompt_id,
            "workflow": workflow_name,
            "status": "completed",
            "outputs": result,
        }

    async def _list_workflows(self) -> List[str]:
        """List available workflow files."""
        workflows_dir = Path(os.getenv("COMFYUI_WORKFLOWS_DIR", "./workflows"))
        if not workflows_dir.exists():
            return []

        return [f.name for f in workflows_dir.glob("*.json")]

    async def _get_image(self, prompt_id: str) -> Dict[str, Any]:
        """Retrieve a generated image."""
        history = await self.client.get_history(1, prompt_id)
        if not history:
            raise ValueError(f"No history found for prompt_id: {prompt_id}")

        # Extract image info from history
        # This will vary based on ComfyUI output format
        # For now, return the raw history
        return history

    def _build_default_workflow(self, **kwargs) -> Dict[str, Any]:
        """Build a basic txt2img workflow."""
        # This is a simplified example - actual workflow structure
        # depends on ComfyUI version and installed nodes
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": kwargs.get("prompt", ""),
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": kwargs.get("negative_prompt", ""),
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": kwargs.get("width", 512),
                    "height": kwargs.get("height", 512),
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": kwargs.get("seed", -1),
                    "steps": kwargs.get("steps", 20),
                    "cfg": kwargs.get("cfg_scale", 7.0),
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "mcp_comfyui",
                    "images": ["6", 0]
                }
            }
        }

    async def run(self):
        """Run the MCP server."""
        # Initialize ComfyUI client
        host = os.getenv("COMFYUI_HOST", "localhost")
        port = int(os.getenv("COMFYUI_PORT", "8188"))
        self.client = ComfyUIClient(host, port)

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-comfyui",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Main entry point."""
    server = ComfyUIServer()
    await server.run()
